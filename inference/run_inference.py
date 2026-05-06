"""
run_inference.py — End-to-end Graph of Thoughts inference loop.

Loads a (possibly LoRA-adapted) causal LM and runs it against gold traces
with the State Executor in the loop: at each step, the model predicts the
next operation, we apply it via StateExecutor, and we re-inject the
ground-truth state into the next prompt.

Output JSON (one entry per input sample)::

    {
        "graph": "<linearized>",
        "algorithm": "bfs",
        "source": 3,
        "gold_steps":     [ {...}, {...} ],
        "predicted_steps":[
            { "step": 0, "operation_predicted": "enqueue(3)",
              "applied": true,  "state": {...}, "induced_subgraph": [...] },
            { "step": 1, "operation_predicted": "visit(3)",
              "applied": false, "error": "Unknown operation: ..." },
            ...
        ]
    }

Usage
-----
python -m inference.run_inference \\
    --model meta-llama/Llama-3.2-1B-Instruct \\
    --adapter checkpoints/llama-3.2-1b-got/final \\
    --trace data/traces/smoke_bfs.json \\
    --limit 5 \\
    --out out/smoke_pred.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.prompt_forcing import (
    build_few_shot_messages,
    build_partial_completion,
    extract_operation,
    format_prompt,
)
from solvers.state_executor import StateExecutor
from training.negative_sampling import CORRECTION_TOKEN
from evaluation.metrics.operation_normalize import operations_match

# torch / transformers / peft are imported lazily inside load_model and
# generate_next_op so that this module can be imported in environments
# without the ML stack (e.g. for scaffolding tests).


# ---------------------------------------------------------------------------
# Graph reconstruction
# ---------------------------------------------------------------------------

_WEIGHTED_EDGE = re.compile(r"\((\d+)\)-\[([\d.\-]+)\]->\((\d+)\)")
_UNWEIGHTED_EDGE = re.compile(r"\((\d+)\)->\((\d+)\)")


def reconstruct_graph(graph_str: str) -> nx.Graph:
    """
    Rebuild an undirected NetworkX graph from a linearized edge string.

    The generators in `data/generators/` produce undirected graphs, linearized
    as `(min_u)-[w]->(max_v)` (weighted) or `(min_u)->(max_v)` (unweighted).
    """
    G = nx.Graph()

    # Weighted edges first — they need weight metadata
    seen = set()
    for m in _WEIGHTED_EDGE.finditer(graph_str):
        u, w_raw, v = int(m.group(1)), m.group(2), int(m.group(3))
        try:
            w = int(w_raw)
        except ValueError:
            w = float(w_raw)
        G.add_edge(u, v, weight=w)
        seen.add((u, v))

    # Unweighted — only add if not already present as weighted
    for m in _UNWEIGHTED_EDGE.finditer(graph_str):
        u, v = int(m.group(1)), int(m.group(2))
        if (u, v) in seen or (v, u) in seen:
            continue
        if G.has_edge(u, v):
            continue
        G.add_edge(u, v)

    return G


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_peft_adapter_dir(path: Optional[str]) -> Optional[str]:
    """Locate a directory containing ``adapter_config.json`` for PEFT load.

    Tries, in order: ``path``; ``path/dagger``; newest ``path/checkpoint-*`` (HF
    Trainer); any immediate subdirectory with ``adapter_config.json``.
    """
    from pathlib import Path

    if not path:
        return path
    p = Path(path)
    if not p.is_dir():
        return path

    def _has_cfg(d: Path) -> bool:
        return d.is_dir() and (d / "adapter_config.json").is_file()

    if _has_cfg(p):
        return str(p)
    if _has_cfg(p / "dagger"):
        return str(p / "dagger")
    checkpoints = sorted(
        p.glob("checkpoint-*"),
        key=lambda x: x.name,
        reverse=True,
    )
    for cp in checkpoints:
        if _has_cfg(cp):
            return str(cp)
    for sub in sorted(p.iterdir()):
        if sub.name.startswith("."):
            continue
        if _has_cfg(sub):
            return str(sub)
    return path


def load_model(
    model_id: str,
    adapter: Optional[str] = None,
    device: str = "auto",
    dtype=None,
    dagger_adapter: Optional[str] = None,
):
    """Load tokenizer + model (optionally with one or two stacked LoRA adapters).

    adapter:        SFT LoRA adapter path (loaded as "sft").
    dagger_adapter: DAgger LoRA adapter path (loaded as "dagger", stacked on top of SFT).
                    Requires adapter to be set.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Match training: CORRECTION_TOKEN is an added special token.
    if CORRECTION_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [CORRECTION_TOKEN]}
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    adapter_resolved = _resolve_peft_adapter_dir(adapter)
    dagger_resolved = _resolve_peft_adapter_dir(dagger_adapter)

    if adapter_resolved and dagger_resolved:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_resolved, adapter_name="sft")
        model.load_adapter(dagger_resolved, adapter_name="dagger")
        try:
            model.set_adapter(["sft", "dagger"])
        except TypeError:
            # Older PEFT rejects list composition here; dagger alone is stacked on frozen SFT.
            model.set_adapter("dagger")
    elif adapter_resolved:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_resolved)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_next_op(
    model,
    tokenizer,
    messages: list[dict],
    partial_text: str,
    correction_id: Optional[int],
    max_new_tokens: int = 48,
) -> tuple[str, bool]:
    """
    Generate one step's text. Returns (first_line_operation, saw_correction_token).

    We concatenate the chat-template prompt + our partial assistant text, then
    greedy-decode. The model is expected to complete `Step <t>: <op>\\n...`.
    We return just the operation on the first line.
    """
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_input = prompt + partial_text
    inputs = tokenizer(full_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    saw_correction = (correction_id is not None) and (correction_id in new_ids)
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    op = extract_operation(text)
    return op, saw_correction


# ---------------------------------------------------------------------------
# Per-sample inference
# ---------------------------------------------------------------------------

def run_one_sample(
    sample: dict,
    model,
    tokenizer,
    correction_id: Optional[int],
    max_steps: Optional[int] = None,
    teacher_forced: bool = True,
    demos: Optional[list[dict]] = None,
    verbose: bool = False,
    dagger_fallback: bool = False,
) -> list[dict]:
    """
    Run GoT inference on one (graph, algorithm, source) sample.

    Two modes:
      • teacher_forced=True (default): the state executor is always advanced
        with the GOLD op, so every prompt contains the correct state. The
        model's prediction is logged but does not drive state. Use this to
        measure pure next-operation accuracy without error compounding.
      • teacher_forced=False: the executor is advanced with the MODEL's op
        (when valid); an invalid op terminates the run. Use this for
        free-running traces where you want to measure drift.

    dagger_fallback: only active when teacher_forced=False. On an invalid op,
        instead of stopping the run, the gold op is silently applied so the
        state executor stays live and subsequent steps can still be collected.
        Set to True when gathering DAgger recovery examples.
    """
    graph_str = sample["graph"]
    algorithm = sample["algorithm"]
    source = sample["source"]
    gold_steps = sample["steps"]
    gold_len = len(gold_steps)

    G = reconstruct_graph(graph_str)
    executor = StateExecutor(G, source)

    if demos:
        messages = build_few_shot_messages(graph_str, algorithm, source, demos)
    else:
        messages = format_prompt(graph_str, algorithm, source)

    budget = max_steps if max_steps is not None else gold_len

    predicted: list[dict] = []
    for t in range(budget):
        partial = build_partial_completion(executor.trace)
        op, saw_correction = generate_next_op(
            model, tokenizer, messages, partial, correction_id
        )

        entry: dict = {"step": t, "operation_predicted": op}
        if saw_correction:
            entry["saw_correction"] = True

        gold_op = gold_steps[t]["operation"] if t < gold_len else None
        if gold_op is not None:
            entry["operation_gold"] = gold_op
            entry["match"] = operations_match(
                op, gold_op, algorithm=algorithm
            )

        if op == CORRECTION_TOKEN or saw_correction:
            entry["applied"] = False
            entry["reason"] = "correction_token"
            predicted.append(entry)
            if verbose:
                print(f"  [{t}] CORRECTION (gold={gold_op})")
            continue

        if teacher_forced:
            # Advance state with the GOLD op regardless of model prediction.
            if gold_op is None:
                break  # ran past gold — nothing to force
            record = executor.apply(gold_op)
            entry["applied"] = True
            entry["operation"] = gold_op  # state advanced via gold
            entry["state"] = record["state"]
            entry["induced_subgraph"] = record["induced_subgraph"]
            predicted.append(entry)
            if verbose:
                marker = "✓" if entry.get("match") else "✗"
                print(f"  [{t}] {marker} pred={op!r}  gold={gold_op!r}")
            continue

        # Free-running mode: apply the model's op.
        try:
            record = executor.apply(op)
            entry["applied"] = True
            entry["operation"] = op
            entry["state"] = record["state"]
            entry["induced_subgraph"] = record["induced_subgraph"]
        except (ValueError, IndexError, KeyError) as e:
            entry["applied"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
            predicted.append(entry)
            if verbose:
                print(f"  [{t}] INVALID op={op!r}: {e}")
            if dagger_fallback and gold_op is not None:
                # Apply gold op so the state executor stays live for later steps.
                executor.apply(gold_op)
                if verbose:
                    print(f"  [{t}] dagger fallback → applied gold={gold_op!r}")
                continue
            break

        predicted.append(entry)
        if verbose:
            print(f"  [{t}] {op}")

    return predicted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run GoT state-executor-in-the-loop inference on a gold trace file."
    )
    ap.add_argument("--model", required=True,
                    help="HF model id or local path to the base causal LM.")
    ap.add_argument("--adapter", default=None,
                    help="Optional SFT LoRA adapter directory (PEFT).")
    ap.add_argument("--dagger-adapter", default=None,
                    help="Optional DAgger LoRA adapter directory to stack on top of --adapter.")
    ap.add_argument("--trace", required=True,
                    help="Path to a JSON gold trace file (array of trace dicts).")
    ap.add_argument("--out", required=True,
                    help="Output JSON path for predictions.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N samples (smoke testing).")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override step budget (default: match gold length).")
    ap.add_argument("--device", default="auto",
                    help="HF device_map value ('auto', 'cpu', 'mps', 'cuda', ...).")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--free-running", action="store_true",
                    help="Apply the model's (possibly wrong) op to the state "
                         "executor. Default is teacher-forced: gold ops drive "
                         "state while the model's prediction is scored.")
    ap.add_argument("--few-shot", type=int, default=0,
                    help="Prepend N demo (user, assistant) pairs sampled from "
                         "the trace file before the target query. Teaches the "
                         "output grammar in-context so untrained base models "
                         "can be evaluated for reasoning ability (default: 0).")
    ap.add_argument("--few-shot-seed", type=int, default=42,
                    help="RNG seed for demo sampling (default: 42).")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    import torch

    args = parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    desc = args.model
    if args.adapter:
        desc += f" + sft={args.adapter}"
    if args.dagger_adapter:
        desc += f" + dagger={args.dagger_adapter}"
    print(f"Loading model {desc}")
    model, tokenizer = load_model(
        args.model, args.adapter, args.device, dtype_map[args.dtype],
        dagger_adapter=args.dagger_adapter,
    )

    correction_id = tokenizer.convert_tokens_to_ids(CORRECTION_TOKEN)
    if correction_id == tokenizer.unk_token_id:
        correction_id = None

    with open(args.trace) as f:
        dataset = json.load(f)

    samples = dataset[: args.limit] if args.limit else dataset

    # Demo sampling for few-shot. Demos come from the full dataset, excluding
    # the current target sample so we never leak the answer.
    import random
    demo_rng = random.Random(args.few_shot_seed)

    predictions = []
    for i, sample in enumerate(samples):
        if args.verbose:
            tag = f"few-shot={args.few_shot}" if args.few_shot else "zero-shot"
            print(f"[{i + 1}/{len(samples)}] {sample['algorithm']} "
                  f"source={sample['source']} gold_len={len(sample['steps'])} ({tag})")

        demos = None
        if args.few_shot > 0:
            # Build a pool from the full dataset, drop the current target.
            target_id = id(sample)
            pool = [s for s in dataset if id(s) != target_id]
            k = min(args.few_shot, len(pool))
            if k < args.few_shot and args.verbose:
                print(f"  [warn] requested {args.few_shot} demos, only {k} available")
            demos = demo_rng.sample(pool, k) if k > 0 else None

        pred_trace = run_one_sample(
            sample, model, tokenizer, correction_id,
            max_steps=args.max_steps,
            teacher_forced=not args.free_running,
            demos=demos,
            verbose=args.verbose,
        )
        predictions.append({
            "graph": sample["graph"],
            "algorithm": sample["algorithm"],
            "source": sample["source"],
            "gold_steps": sample["steps"],
            "predicted_steps": pred_trace,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
