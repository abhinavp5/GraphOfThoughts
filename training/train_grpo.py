"""
Train Graph-of-Thoughts models with GRPO-style reinforcement updates.

Usage
-----
python training/train_grpo.py --config training/configs/llama_3_1b.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
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
from training.teacher_forcing import format_prompt
from training.sft import (
    _apply_hf_hub_cache,
    _maybe_warn_hf_gated_model,
    setup_model,
    setup_tokenizer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("got.grpo")


@dataclass
class GRPOSample:
    prompt: str
    gold_ops: list[str]


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


def build_grpo_dataset(
    traces: list[dict],
    tokenizer,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[GRPOSample]:
    rng = random.Random(seed)
    ordered = traces[:]
    rng.shuffle(ordered)
    if max_samples is not None:
        ordered = ordered[:max_samples]

    dataset: list[GRPOSample] = []
    for trace in ordered:
        prompt = _messages_to_prompt_text(tokenizer, trace)
        gold_ops = [s["operation"] for s in trace["steps"]]
        dataset.append(GRPOSample(prompt=prompt, gold_ops=gold_ops))
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


def _extract_ops(text: str) -> list[str]:
    ops = []
    for line in text.splitlines():
        m = re.match(r"^\s*Step\s+\d+\s*:\s*(.+?)\s*$", line)
        if m:
            ops.append(m.group(1))
    return ops


def _trace_reward(pred_text: str, gold_ops: list[str]) -> float:
    pred_ops = _extract_ops(pred_text)
    if not pred_ops:
        return -1.0
    matches = 0
    for idx, op in enumerate(pred_ops):
        if idx >= len(gold_ops):
            break
        if op == gold_ops[idx]:
            matches += 1
        else:
            break
    coverage = min(len(pred_ops), len(gold_ops)) / max(1, len(gold_ops))
    return (matches / max(1, len(gold_ops))) + 0.1 * coverage


def run_grpo_stage(
    model,
    ref_model,
    tokenizer,
    train_data: list[GRPOSample],
    cfg: dict,
    max_length: int,
) -> None:
    grpo_cfg = cfg.get("grpo", {})
    epochs = int(grpo_cfg.get("epochs", 1))
    lr = float(grpo_cfg.get("learning_rate", 3e-6))
    group_size = int(grpo_cfg.get("group_size", 4))
    max_new_tokens = int(grpo_cfg.get("max_new_tokens", 256))
    temperature = float(grpo_cfg.get("temperature", 0.8))
    kl_coef = float(grpo_cfg.get("kl_coef", 0.02))
    grad_clip = float(grpo_cfg.get("grad_clip", 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logger.info(
        "Starting GRPO (epochs=%s, group_size=%s, lr=%s)",
        epochs,
        group_size,
        lr,
    )
    for epoch in range(epochs):
        random.shuffle(train_data)
        for i, sample in enumerate(train_data, start=1):
            prompt_ids = tokenizer(
                sample.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).input_ids.to(model.device)

            rewards = []
            responses = []
            with torch.no_grad():
                for _ in range(group_size):
                    out = model.generate(
                        input_ids=prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen_ids = out[:, prompt_ids.shape[1]:]
                    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    responses.append(text)
                    rewards.append(_trace_reward(text, sample.gold_ops))

            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.device)
            advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-6)

            policy_losses = []
            kl_terms = []
            for response, adv in zip(responses, advantages):
                pi_logp = _sequence_logprob(model, tokenizer, sample.prompt, response, max_length=max_length)
                ref_logp = _no_grad_logprob(ref_model, tokenizer, sample.prompt, response, max_length=max_length)
                policy_losses.append(-adv * pi_logp.mean())
                kl_terms.append((pi_logp - ref_logp).mean())

            loss = torch.stack(policy_losses).mean() + kl_coef * torch.stack(kl_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if i % 10 == 0:
                logger.info(
                    "epoch %d step %d/%d loss=%.4f reward_mean=%.4f",
                    epoch + 1,
                    i,
                    len(train_data),
                    float(loss.item()),
                    float(reward_tensor.mean().item()),
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO trainer for Graph-of-Thoughts traces.")
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
    grpo_train = build_grpo_dataset(
        train_traces,
        tokenizer,
        max_samples=args.max_train_samples,
        seed=seed,
    )
    logger.info("Loaded %d GRPO samples", len(grpo_train))

    max_length = int(data_cfg.get("max_seq_length", model_cfg.get("max_seq_length", 2048)))
    run_grpo_stage(model, ref_model, tokenizer, grpo_train, cfg, max_length=max_length)

    output_dir = args.output_dir or cfg.get("training", {}).get("output_dir", "checkpoints/got-grpo")
    output_dir = os.path.join(output_dir, "grpo")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved GRPO model to %s", output_dir)


if __name__ == "__main__":
    main()
