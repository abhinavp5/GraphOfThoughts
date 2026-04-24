"""
State consistency metric for GoT prediction traces.

Measures alignment between model-applied state transitions and deterministic
gold states/subgraphs. Most informative in free-running mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _canon_state(s: dict | None) -> dict:
    s = s or {}
    return {
        "visited": list(s.get("visited", [])),
        "frontier": list(s.get("frontier", [])),
        "distances": {str(k): v for k, v in s.get("distances", {}).items()},
        "parent": {str(k): (str(v) if v is not None else None) for k, v in s.get("parent", {}).items()},
    }


def _canon_subgraph(sub: list | None) -> set[tuple]:
    out: set[tuple] = set()
    for e in (sub or []):
        if len(e) >= 3:
            u, v, w = e[0], e[1], e[2]
        elif len(e) == 2:
            u, v, w = e[0], e[1], None
        else:
            continue
        out.add((int(u), int(v), w))
    return out


def score(predictions: list[dict]) -> dict:
    total_comparable = 0
    state_matches = 0
    subgraph_matches = 0
    both_matches = 0
    per_sample = []

    for sample in predictions:
        gold = sample.get("gold_steps", [])
        pred = sample.get("predicted_steps", [])
        pred_by_step = {int(p.get("step", -1)): p for p in pred if p.get("applied") and "state" in p}

        sample_total = 0
        sample_state = 0
        sample_sub = 0
        sample_both = 0

        for t, g in enumerate(gold):
            p = pred_by_step.get(t)
            if not p:
                continue
            sample_total += 1
            total_comparable += 1

            state_ok = _canon_state(p.get("state")) == _canon_state(g.get("state"))
            sub_ok = _canon_subgraph(p.get("induced_subgraph")) == _canon_subgraph(g.get("induced_subgraph"))
            if state_ok:
                sample_state += 1
                state_matches += 1
            if sub_ok:
                sample_sub += 1
                subgraph_matches += 1
            if state_ok and sub_ok:
                sample_both += 1
                both_matches += 1

        per_sample.append(
            {
                "algorithm": sample.get("algorithm"),
                "source": sample.get("source"),
                "comparable_steps": sample_total,
                "state_consistency": (sample_state / sample_total) if sample_total else 0.0,
                "subgraph_consistency": (sample_sub / sample_total) if sample_total else 0.0,
                "joint_consistency": (sample_both / sample_total) if sample_total else 0.0,
            }
        )

    return {
        "n_samples": len(predictions),
        "comparable_steps": total_comparable,
        "state_consistency": (state_matches / total_comparable) if total_comparable else 0.0,
        "subgraph_consistency": (subgraph_matches / total_comparable) if total_comparable else 0.0,
        "joint_consistency": (both_matches / total_comparable) if total_comparable else 0.0,
        "per_sample": per_sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="Path to run_inference output JSON.")
    ap.add_argument("--out", default=None, help="Optional output JSON path.")
    args = ap.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)
    metrics = score(predictions)

    print(f"Samples:               {metrics['n_samples']}")
    print(f"Comparable steps:      {metrics['comparable_steps']}")
    print(f"State consistency:     {metrics['state_consistency']:.4f}")
    print(f"Subgraph consistency:  {metrics['subgraph_consistency']:.4f}")
    print(f"Joint consistency:     {metrics['joint_consistency']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote state consistency to {args.out}")


if __name__ == "__main__":
    main()
