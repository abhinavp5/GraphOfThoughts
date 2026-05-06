"""
Operation accuracy — the headline metric.

Given a predictions file (list of {gold_steps, predicted_steps}), computes:

  • per-step accuracy:   P(pred_op_t == gold_op_t)
  • trace accuracy:      fraction of traces where all steps match
  • first-error depth:   average step index of the first mismatch
  • coverage:            fraction of gold steps the model attempted
                         (drops if the model stopped early / hit an invalid op)

Public API
----------
score(predictions)               → dict of aggregate metrics
score_sample(sample)             → dict of per-sample metrics
per_step_matches(sample)         → list[bool]  (length = len(gold_steps))

CLI
---
python -m evaluation.metrics.operation_accuracy predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics.operation_normalize import operations_match


def per_step_matches(sample: dict) -> list[bool]:
    """
    Return list of booleans aligned to gold_steps: True iff the model
    predicted the exact same operation at that step.

    Steps beyond the predicted trace (e.g., after an invalid op) count as
    misses (False).
    """
    gold = sample["gold_steps"]
    pred = sample.get("predicted_steps", [])
    algo = sample.get("algorithm")

    # Index predicted by step number so we tolerate CORRECTION steps etc.
    pred_by_step: dict[int, dict] = {}
    for p in pred:
        # Skip correction placeholders (not actual step predictions)
        if p.get("reason") == "correction_token":
            continue
        pred_by_step.setdefault(p["step"], p)

    matches = []
    for t, g in enumerate(gold):
        p = pred_by_step.get(t)
        if p is None:
            matches.append(False)
            continue
        matches.append(
            operations_match(
                p.get("operation_predicted"),
                g.get("operation"),
                algorithm=algo,
            )
        )
    return matches


def score_sample(sample: dict) -> dict:
    """Per-sample metrics."""
    matches = per_step_matches(sample)
    n = len(matches)
    n_correct = sum(matches)
    first_error = next((i for i, m in enumerate(matches) if not m), None)

    pred_count = sum(
        1 for p in sample.get("predicted_steps", [])
        if p.get("reason") != "correction_token"
    )

    return {
        "n_steps": n,
        "n_correct": n_correct,
        "step_accuracy": (n_correct / n) if n else 0.0,
        "trace_correct": (n_correct == n and n > 0),
        "first_error_step": first_error,
        "coverage": (min(pred_count, n) / n) if n else 0.0,
    }


def score(predictions: list[dict]) -> dict:
    """Aggregate metrics across a predictions file."""
    per_sample = [score_sample(s) for s in predictions]

    total_steps = sum(s["n_steps"] for s in per_sample)
    total_correct = sum(s["n_correct"] for s in per_sample)
    traces_correct = sum(1 for s in per_sample if s["trace_correct"])

    first_errors = [s["first_error_step"] for s in per_sample
                    if s["first_error_step"] is not None]
    mean_first_error = (sum(first_errors) / len(first_errors)) if first_errors else None

    mean_coverage = (
        sum(s["coverage"] for s in per_sample) / len(per_sample)
        if per_sample else 0.0
    )

    return {
        "n_samples": len(per_sample),
        "total_steps": total_steps,
        "total_correct": total_correct,
        "step_accuracy": (total_correct / total_steps) if total_steps else 0.0,
        "trace_accuracy": (traces_correct / len(per_sample)) if per_sample else 0.0,
        "mean_first_error_step": mean_first_error,
        "mean_coverage": mean_coverage,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="Path to run_inference.py output JSON.")
    ap.add_argument("--out", default=None,
                    help="Optional: write full metrics JSON here.")
    args = ap.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    metrics = score(predictions)

    print(f"Samples:          {metrics['n_samples']}")
    print(f"Total steps:      {metrics['total_steps']}")
    print(f"Step accuracy:    {metrics['step_accuracy']:.4f}")
    print(f"Trace accuracy:   {metrics['trace_accuracy']:.4f}")
    if metrics["mean_first_error_step"] is not None:
        print(f"Mean first-error step: {metrics['mean_first_error_step']:.2f}")
    print(f"Mean coverage:    {metrics['mean_coverage']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nWrote metrics to {args.out}")


if __name__ == "__main__":
    main()
