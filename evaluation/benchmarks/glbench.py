"""
GLBench integration for Graph-of-Thoughts evaluation.

This adapter normalizes GLBench-style JSON records into GoT trace format,
runs inference, and computes:
  - operation_accuracy metrics
  - failure_analysis metrics

Example
-------
python -m evaluation.benchmarks.glbench \
  --input data/benchmarks/glbench_eval.json \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out-prefix out/glbench_qwen
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.benchmarks.common import dump_json, load_json, normalize_benchmark_record
from evaluation.metrics.operation_accuracy import score as score_operation_accuracy
from evaluation.metrics.failure_analysis import analyze as analyze_failures
from inference.run_inference import load_model, run_one_sample
from training.negative_sampling import CORRECTION_TOKEN


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="GLBench JSON file path.")
    ap.add_argument("--model", required=True, help="HF model id or local model path.")
    ap.add_argument("--adapter", default=None, help="Optional SFT LoRA adapter directory.")
    ap.add_argument("--dagger-adapter", default=None, help="Optional DAgger LoRA adapter to stack on top of --adapter.")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (no extension).")
    ap.add_argument("--limit", type=int, default=None, help="Run first N records.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--free-running", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--algorithm",
        default="bfs",
        choices=["bfs", "dfs", "dijkstra"],
        help="Evaluate only GLBench rows whose query_type/algorithm matches this task.",
    )
    return ap.parse_args()


def normalize_glbench(records: list[dict]) -> list[dict]:
    # Common GLBench aliases found in graph reasoning exports.
    graph_keys = ["graph", "edge_text", "graph_text", "graph_linearized", "input_graph"]
    algorithm_keys = ["algorithm", "query_type", "task", "task_name"]
    source_keys = ["source", "start", "query_node", "start_node"]
    steps_keys = ["steps", "gold_steps", "operations", "gold_operations", "target_operations"]

    out = []
    for rec in records:
        out.append(
            normalize_benchmark_record(
                rec,
                graph_keys=graph_keys,
                algorithm_keys=algorithm_keys,
                source_keys=source_keys,
                steps_keys=steps_keys,
            )
        )
    return out


def main() -> None:
    import torch

    args = parse_args()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    raw = load_json(args.input)
    samples = normalize_glbench(raw)
    want = str(args.algorithm).lower().strip()
    samples = [s for s in samples if s.get("algorithm") == want]
    if not samples:
        raise SystemExit(
            f"[glbench] No {want} records after normalization/filter. "
            "Check query_type / target_operations and graph fields in the JSON."
        )
    if args.limit:
        samples = samples[: args.limit]

    model, tokenizer = load_model(
        args.model,
        adapter=args.adapter,
        device=args.device,
        dtype=dtype_map[args.dtype],
        dagger_adapter=args.dagger_adapter,
    )

    correction_id = tokenizer.convert_tokens_to_ids(CORRECTION_TOKEN)
    if correction_id == tokenizer.unk_token_id:
        correction_id = None

    predictions = []
    for sample in samples:
        pred_steps = run_one_sample(
            sample,
            model,
            tokenizer,
            correction_id,
            max_steps=args.max_steps,
            teacher_forced=not args.free_running,
            demos=None,
            verbose=False,
        )
        predictions.append(
            {
                "graph": sample["graph"],
                "algorithm": sample["algorithm"],
                "source": sample["source"],
                "gold_steps": sample["steps"],
                "predicted_steps": pred_steps,
                "benchmark_meta": sample.get("benchmark_meta", {}),
            }
        )

    out_prefix = Path(args.out_prefix)
    pred_path = str(out_prefix) + "_predictions.json"
    op_path = str(out_prefix) + "_operation_accuracy.json"
    fail_path = str(out_prefix) + "_failure_analysis.json"

    dump_json(pred_path, predictions)
    dump_json(op_path, score_operation_accuracy(predictions))
    dump_json(fail_path, analyze_failures(predictions))

    print(f"Wrote predictions:       {pred_path}")
    print(f"Wrote operation metrics: {op_path}")
    print(f"Wrote failure analysis:  {fail_path}")


if __name__ == "__main__":
    main()
