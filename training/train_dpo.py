"""
Train Graph-of-Thoughts models with DPO.

Usage
-----
python training/train_dpo.py --config training/configs/llama_3_1b.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dataset import load_traces
from training.teacher_forcing import format_prompt, format_trace_completion
from training.sft import (
    _apply_hf_hub_cache,
    _maybe_warn_hf_gated_model,
    setup_model,
    setup_tokenizer,
)
from training.negative_sampling import corrupt_trace


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("got.dpo")


@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str


def _messages_to_prompt_text(tokenizer, trace: dict) -> str:
    messages = format_prompt(
        graph_str=trace["graph"],
        algorithm=trace["algorithm"],
        source=trace["source"],
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_preference_dataset(
    traces: list[dict],
    tokenizer,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[PreferenceSample]:
    rng = random.Random(seed)
    ordered = traces[:]
    rng.shuffle(ordered)
    if max_samples is not None:
        ordered = ordered[:max_samples]

    dataset: list[PreferenceSample] = []
    for trace in ordered:
        prompt = _messages_to_prompt_text(tokenizer, trace)
        chosen = format_trace_completion(trace["steps"])
        rejected_trace = corrupt_trace(trace, rng=rng)
        rejected = format_trace_completion(rejected_trace["steps"])
        dataset.append(PreferenceSample(prompt=prompt, chosen=chosen, rejected=rejected))
    return dataset


def _sequence_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> torch.Tensor:
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)
    full_ids = tokenizer(
        prompt + completion,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    ).input_ids.to(model.device)
    labels = full_ids.clone()
    prompt_len = min(prompt_ids.shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target != -100
    safe_target = target.masked_fill(~mask, 0)
    tok_logp = F.log_softmax(logits, dim=-1).gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(dim=1)


def _no_grad_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> torch.Tensor:
    with torch.no_grad():
        return _sequence_logprob(model, tokenizer, prompt, completion, max_length=max_length)


def run_dpo_stage(
    model,
    ref_model,
    tokenizer,
    train_data: list[PreferenceSample],
    cfg: dict,
    max_length: int,
) -> None:
    dpo_cfg = cfg.get("dpo", {})
    beta = float(dpo_cfg.get("beta", 0.1))
    epochs = int(dpo_cfg.get("epochs", 1))
    lr = float(dpo_cfg.get("learning_rate", 5e-6))
    grad_clip = float(dpo_cfg.get("grad_clip", 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logger.info("Starting DPO (epochs=%s, beta=%s, lr=%s)", epochs, beta, lr)
    model.train()
    for epoch in range(epochs):
        random.shuffle(train_data)
        running = 0.0
        for i, sample in enumerate(train_data, start=1):
            pi_chosen = _sequence_logprob(model, tokenizer, sample.prompt, sample.chosen, max_length=max_length)
            pi_rejected = _sequence_logprob(model, tokenizer, sample.prompt, sample.rejected, max_length=max_length)
            ref_chosen = _no_grad_logprob(ref_model, tokenizer, sample.prompt, sample.chosen, max_length=max_length)
            ref_rejected = _no_grad_logprob(ref_model, tokenizer, sample.prompt, sample.rejected, max_length=max_length)

            delta = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
            loss = -F.logsigmoid(beta * delta).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running += float(loss.item())
            if i % 20 == 0:
                logger.info(
                    "epoch %d step %d/%d loss=%.4f",
                    epoch + 1,
                    i,
                    len(train_data),
                    running / i,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO trainer for Graph-of-Thoughts traces.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap number of training traces.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    lora_cfg = cfg["lora"]

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    _apply_hf_hub_cache(cfg)
    _maybe_warn_hf_gated_model(model_cfg["name"])

    tokenizer = setup_tokenizer(model_cfg["name"], special_tokens=cfg.get("special_tokens"))
    model = setup_model(model_cfg["name"], tokenizer, lora_cfg)
    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=True,
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    train_traces = load_traces(
        data_cfg.get("train_dir", "data/traces"),
        data_cfg.get("train_pattern", "train_*.json"),
    )
    pref_train = build_preference_dataset(
        train_traces,
        tokenizer,
        max_samples=args.max_train_samples,
        seed=seed,
    )
    logger.info("Loaded %d preference samples", len(pref_train))

    max_length = int(data_cfg.get("max_seq_length", model_cfg.get("max_seq_length", 2048)))
    run_dpo_stage(model, ref_model, tokenizer, pref_train, cfg, max_length=max_length)

    output_dir = args.output_dir or cfg.get("training", {}).get("output_dir", "checkpoints/got-dpo")
    output_dir = os.path.join(output_dir, "dpo")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved DPO model to %s", output_dir)


if __name__ == "__main__":
    main()
