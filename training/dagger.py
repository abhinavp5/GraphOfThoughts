"""
Minimal DAgger-style stage-2 pipeline for graph algorithms.

Two stages:
  1) collect: run free-running rollouts and extract (incorrect_state -> gold_recovery_op)
  2) finetune: supervised train on collected recovery examples
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from inference.run_inference import load_model, run_one_sample
from training.dataset import load_traces
from training.negative_sampling import CORRECTION_TOKEN


_ALGO_SPECS: dict[str, dict[str, object]] = {
    "bfs": {
        "name": "BFS",
        "ops": [
            "enqueue(node)            — add node to the back of the queue",
            "dequeue(node)            — remove node from the front of the queue",
            "mark_visited(node)       — record node as visited",
            "set_parent(child, parent)— record that child was reached from parent",
        ],
    },
    "dfs": {
        "name": "DFS",
        "ops": [
            "push(node)               — push node onto the stack",
            "pop(node)                — pop node from the stack",
            "visit(node)              — record node as visited",
            "set_parent(child, parent)— record that child was reached from parent",
        ],
    },
    "dijkstra": {
        "name": "DIJKSTRA",
        "ops": [
            "init_source(node)        — set distance(node)=0 and enqueue it",
            "settle(node)             — mark node's distance as final",
            "relax(u, v, new_dist)    — update distance(v) via u and record predecessor",
        ],
    },
}


def _build_recovery_prompt(sample: dict, state: dict, wrong_op: str) -> str:
    algo = str(sample.get("algorithm", "")).strip().lower()
    if algo not in _ALGO_SPECS:
        raise ValueError(f"Unknown sample['algorithm']={sample.get('algorithm')!r}; expected one of {sorted(_ALGO_SPECS)}")
    spec = _ALGO_SPECS[algo]
    op_lines = "\n".join(f"  {line}" for line in spec["ops"])  # type: ignore[index]
    return (
        f"You are a graph algorithm executor for {spec['name']}.\n"
        f"Graph: {sample['graph']}\n"
        f"Algorithm: {spec['name']}\n"
        f"Source: {sample['source']}\n"
        "At each step, output exactly one operation from the following set:\n"
        f"{op_lines}\n"
        "No other operation names are permitted.\n"
        f"Current state: visited={state.get('visited', [])} "
        f"frontier={state.get('frontier', [])} distances={state.get('distances', {})} "
        f"parent={state.get('parent', {})}\n"
        f"Previous wrong operation: {wrong_op if wrong_op else '(no output)'}\n"
        "Output exactly one line as the recovery operation:\n"
    )


def collect(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    random.seed(args.seed)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model, tokenizer = load_model(args.model, args.adapter, args.device, dtype_map[args.dtype])
    correction_id = tokenizer.convert_tokens_to_ids(CORRECTION_TOKEN)
    if correction_id == tokenizer.unk_token_id:
        correction_id = None

    traces = load_traces(args.trace_dir, args.trace_pattern)
    if args.limit:
        traces = traces[: args.limit]

    out = []
    for sample in traces:
        pred = run_one_sample(
            sample,
            model,
            tokenizer,
            correction_id,
            max_steps=args.max_steps,
            teacher_forced=False,
            dagger_fallback=True,  # on invalid op: apply gold, continue collecting
            demos=None,
            verbose=args.verbose,
        )
        gold = sample["steps"]
        for p in pred:
            t = int(p.get("step", -1))
            if t < 0 or t >= len(gold):
                continue
            wrong = p.get("operation_predicted")
            correct = gold[t]["operation"]
            if wrong is None:
                continue
            if wrong == correct:
                continue

            # Option B: include invalid-op failures too.
            # If the wrong op could not be applied, we cannot snapshot a "wrong" state.
            # Instead, use the gold state at step t as the recovery context.
            used_state = p.get("state") if (p.get("applied") and "state" in p) else gold[t].get("state", {})
            prompt = _build_recovery_prompt(sample, used_state, wrong)
            out.append(
                {
                    "graph": sample["graph"],
                    "algorithm": sample["algorithm"],
                    "source": sample["source"],
                    "step": t,
                    "wrong_operation": wrong,
                    "correct_operation": correct,
                    "state": used_state,
                    "wrong_applied": bool(p.get("applied")),
                    "prompt": prompt,
                    "target": correct,
                }
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Collected {len(out)} recovery examples -> {args.out}")


class RecoveryDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_len: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        for r in records:
            full = r["prompt"] + r["target"]
            prompt_enc = tokenizer(r["prompt"], add_special_tokens=False)
            full_enc = tokenizer(full, truncation=True, max_length=max_len, add_special_tokens=False)
            ids = full_enc["input_ids"]
            attn = full_enc["attention_mask"]
            labels = ids[:]
            prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            self.samples.append({"input_ids": ids, "attention_mask": attn, "labels": labels})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {k: torch.tensor(v, dtype=torch.long) for k, v in s.items()}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in batch:
            n = len(x["input_ids"])
            pad = max_len - n
            out["input_ids"].append(torch.cat([x["input_ids"], torch.full((pad,), self.pad_id)]))
            out["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            out["labels"].append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in out.items()}


def finetune(args: argparse.Namespace) -> None:
    with open(args.recovery_json) as f:
        records = json.load(f)
    if not records:
        print("[warn] No recovery examples found — nothing to fine-tune on. Exiting cleanly.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = RecoveryDataset(records, tokenizer, max_len=args.max_seq_len)
    collator = Collator(tokenizer.pad_token_id)
    os.makedirs(args.output_dir, exist_ok=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    # Save only the LoRA adapter — avoids OOM from gathering full 7B weight shards.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DAgger stage-2 LoRA adapter to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal DAgger-style stage-2 training for graph algorithms.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="Collect recovery examples from free-running rollouts.")
    c.add_argument("--model", required=True)
    c.add_argument("--adapter", default=None)
    c.add_argument("--trace-dir", default="data/traces")
    c.add_argument("--trace-pattern", default="train_*.json")
    c.add_argument("--out", required=True)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--max-steps", type=int, default=None)
    c.add_argument("--device", default="auto")
    c.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--verbose", action="store_true", help="Print each step's prediction during rollout.")

    f = sub.add_parser("finetune", help="Fine-tune on collected recovery examples.")
    f.add_argument("--model", required=True)
    f.add_argument("--recovery-json", required=True)
    f.add_argument("--output-dir", required=True)
    f.add_argument("--epochs", type=int, default=1)
    f.add_argument("--batch-size", type=int, default=2)
    f.add_argument("--grad-accum", type=int, default=4)
    f.add_argument("--lr", type=float, default=1e-5)
    f.add_argument("--max-seq-len", type=int, default=1024)
    f.add_argument("--lora-r", type=int, default=16)
    f.add_argument("--lora-alpha", type=int, default=32)
    f.add_argument("--lora-dropout", type=float, default=0.05)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "collect":
        collect(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
