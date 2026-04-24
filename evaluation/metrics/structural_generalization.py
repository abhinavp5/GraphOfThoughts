"""
Structural generalization report for BFS traces.

Groups performance by graph size to quantify transfer from small training
graphs to larger evaluation graphs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluation.metrics.operation_accuracy import per_step_matches

_NODE_RE = re.compile(r"\((\d+)\)")


def _graph_size(graph_str: str) -> int:
    return len({int(m.group(1)) for m in _NODE_RE.finditer(graph_str)})


def score(predictions: list[dict], train_max_n: int) -> dict:
    grouped: dict[str, dict] = {
        "in_distribution": {"steps": 0, "correct": 0, "samples": 0},
        "out_of_distribution": {"steps": 0, "correct": 0, "samples": 0},
    }
    by_n: dict[int, dict] = {}

    for sample in predictions:
        n = _graph_size(sample.get("graph", ""))
        matches = per_step_matches(sample)
        steps = len(matches)
        correct = sum(matches)
        bucket = "in_distribution" if n <= train_max_n else "out_of_distribution"
        grouped[bucket]["steps"] += steps
        grouped[bucket]["correct"] += correct
        grouped[bucket]["samples"] += 1

        if n not in by_n:
            by_n[n] = {"steps": 0, "correct": 0, "samples": 0}
        by_n[n]["steps"] += steps
        by_n[n]["correct"] += correct
        by_n[n]["samples"] += 1

    def _acc(d: dict) -> float:
        return (d["correct"] / d["steps"]) if d["steps"] else 0.0

    return {
        "train_max_n": train_max_n,
        "in_distribution_step_accuracy": _acc(grouped["in_distribution"]),
        "out_of_distribution_step_accuracy": _acc(grouped["out_of_distribution"]),
        "gap_ood_minus_id": _acc(grouped["out_of_distribution"]) - _acc(grouped["in_distribution"]),
        "grouped": grouped,
        "by_graph_size": {
            str(n): {
                **v,
                "step_accuracy": (v["correct"] / v["steps"]) if v["steps"] else 0.0,
            }
            for n, v in sorted(by_n.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="Path to run_inference output JSON.")
    ap.add_argument("--train-max-n", type=int, required=True, help="Max graph size used in training.")
    ap.add_argument("--out", default=None, help="Optional output JSON path.")
    args = ap.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)
    report = score(predictions, train_max_n=args.train_max_n)

    print(f"Train max n:                    {report['train_max_n']}")
    print(f"In-distribution step accuracy:  {report['in_distribution_step_accuracy']:.4f}")
    print(f"Out-of-distribution step acc:   {report['out_of_distribution_step_accuracy']:.4f}")
    print(f"OOD-ID gap:                     {report['gap_ood_minus_id']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote structural generalization report to {args.out}")


if __name__ == "__main__":
    main()
