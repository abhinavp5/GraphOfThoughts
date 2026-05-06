"""
Failure analysis for Graph-of-Thoughts prediction traces.

Given run_inference outputs, this script identifies where/why models fail:
  - first-error depth distribution
  - failure rate by step index and early/mid/late buckets
  - failure rate by gold operation type (e.g., enqueue, visit, relax)
  - top confusion pairs (gold_op -> predicted_op)
  - invalid-op/runtime errors surfaced by StateExecutor

Usage
-----
python -m evaluation.metrics.failure_analysis out/pred_bfs.json
python -m evaluation.metrics.failure_analysis out/pred_bfs.json --out out/failures_bfs.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.metrics.operation_normalize import operations_match


def _op_type(op: str | None) -> str:
    if not op:
        return "<missing>"
    if "(" in op:
        return op.split("(", 1)[0].strip()
    return op.strip()


def _step_bucket(step_idx: int, total_steps: int) -> str:
    if total_steps <= 0:
        return "unknown"
    frac = step_idx / max(1, total_steps - 1)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "middle"
    return "late"


def analyze(predictions: list[dict]) -> dict:
    n_samples = len(predictions)

    total_gold_steps = 0
    total_failures = 0
    total_matches = 0
    first_error_steps: list[int] = []
    samples_with_any_error = 0

    errors_by_type: Counter[str] = Counter()
    op_failures: Counter[str] = Counter()
    op_totals: Counter[str] = Counter()
    bucket_failures: Counter[str] = Counter()
    bucket_totals: Counter[str] = Counter()
    step_failures: Counter[int] = Counter()
    step_totals: Counter[int] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    algorithm_failures: Counter[str] = Counter()
    algorithm_totals: Counter[str] = Counter()

    per_sample = []

    for sample in predictions:
        algo = sample.get("algorithm", "unknown")
        gold_steps = sample.get("gold_steps", [])
        pred_steps = sample.get("predicted_steps", [])
        gold_len = len(gold_steps)

        total_gold_steps += gold_len
        algorithm_totals[algo] += gold_len

        pred_by_step: dict[int, dict] = {}
        for p in pred_steps:
            if p.get("reason") == "correction_token":
                continue
            pred_by_step.setdefault(int(p.get("step", -1)), p)

        sample_failures = 0
        sample_first_error = None
        sample_invalid = 0

        for t, g in enumerate(gold_steps):
            step_totals[t] += 1
            bucket = _step_bucket(t, gold_len)
            bucket_totals[bucket] += 1

            gold_op = g.get("operation")
            gold_type = _op_type(gold_op)
            op_totals[gold_type] += 1

            p = pred_by_step.get(t)
            if p is None:
                # Missing prediction counts as failure.
                fail_type = "missing_prediction"
                errors_by_type[fail_type] += 1
                step_failures[t] += 1
                bucket_failures[bucket] += 1
                op_failures[gold_type] += 1
                confusion[(gold_op or "<missing>", "<none>")] += 1
                total_failures += 1
                algorithm_failures[algo] += 1
                sample_failures += 1
                if sample_first_error is None:
                    sample_first_error = t
                continue

            pred_op = p.get("operation_predicted")
            is_match = operations_match(pred_op, gold_op, algorithm=algo)
            applied = bool(p.get("applied", False))

            if is_match:
                total_matches += 1
            else:
                fail_type = "wrong_operation"
                errors_by_type[fail_type] += 1
                step_failures[t] += 1
                bucket_failures[bucket] += 1
                op_failures[gold_type] += 1
                confusion[(gold_op or "<missing>", pred_op or "<missing>")] += 1
                total_failures += 1
                algorithm_failures[algo] += 1
                sample_failures += 1
                if sample_first_error is None:
                    sample_first_error = t

            # Free-running mode can include explicit executor errors.
            if not applied:
                if p.get("error"):
                    errors_by_type["invalid_or_runtime_error"] += 1
                    sample_invalid += 1
                elif p.get("reason") == "correction_token":
                    errors_by_type["correction_token"] += 1

        if sample_first_error is not None:
            samples_with_any_error += 1
            first_error_steps.append(sample_first_error)

        per_sample.append(
            {
                "algorithm": algo,
                "source": sample.get("source"),
                "n_gold_steps": gold_len,
                "n_failures": sample_failures,
                "step_accuracy": (gold_len - sample_failures) / gold_len if gold_len else 0.0,
                "first_error_step": sample_first_error,
                "n_invalid_or_runtime_errors": sample_invalid,
            }
        )

    op_failure_rates = {
        op: (op_failures[op] / op_totals[op]) if op_totals[op] else 0.0
        for op in sorted(op_totals.keys())
    }
    bucket_failure_rates = {
        b: (bucket_failures[b] / bucket_totals[b]) if bucket_totals[b] else 0.0
        for b in ["early", "middle", "late", "unknown"]
        if bucket_totals[b] > 0
    }
    step_failure_rates = {
        str(step): (step_failures[step] / step_totals[step]) if step_totals[step] else 0.0
        for step in sorted(step_totals.keys())
    }
    algorithm_failure_rates = {
        algo: (algorithm_failures[algo] / algorithm_totals[algo]) if algorithm_totals[algo] else 0.0
        for algo in sorted(algorithm_totals.keys())
    }

    top_confusions = [
        {"gold_operation": g, "predicted_operation": p, "count": c}
        for (g, p), c in confusion.most_common(25)
    ]

    return {
        "n_samples": n_samples,
        "total_gold_steps": total_gold_steps,
        "total_matches": total_matches,
        "total_failures": total_failures,
        "overall_step_accuracy": (total_matches / total_gold_steps) if total_gold_steps else 0.0,
        "overall_failure_rate": (total_failures / total_gold_steps) if total_gold_steps else 0.0,
        "samples_with_any_error": samples_with_any_error,
        "mean_first_error_step": (
            sum(first_error_steps) / len(first_error_steps) if first_error_steps else None
        ),
        "error_type_counts": dict(errors_by_type),
        "failure_rate_by_algorithm": algorithm_failure_rates,
        "failure_rate_by_step_bucket": bucket_failure_rates,
        "failure_rate_by_step_index": step_failure_rates,
        "failure_rate_by_gold_operation_type": op_failure_rates,
        "top_operation_confusions": top_confusions,
        "per_sample": per_sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="Path to run_inference.py output JSON.")
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    report = analyze(predictions)

    print(f"Samples:                 {report['n_samples']}")
    print(f"Total gold steps:        {report['total_gold_steps']}")
    print(f"Overall step accuracy:   {report['overall_step_accuracy']:.4f}")
    print(f"Overall failure rate:    {report['overall_failure_rate']:.4f}")
    if report["mean_first_error_step"] is not None:
        print(f"Mean first-error step:   {report['mean_first_error_step']:.2f}")
    print("\nFailure rate by phase:")
    for k, v in report["failure_rate_by_step_bucket"].items():
        print(f"  {k:>6}: {v:.4f}")

    print("\nTop operation confusions:")
    for row in report["top_operation_confusions"][:10]:
        print(
            "  "
            f"{row['gold_operation']} -> {row['predicted_operation']} "
            f"(n={row['count']})"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote failure analysis to {args.out}")


if __name__ == "__main__":
    main()
