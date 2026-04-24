from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Metric JSON files.")
    ap.add_argument("--labels", nargs="+", required=True, help="Bar labels matching inputs.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    args = ap.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting: pip install matplotlib") from e

    vals = []
    for p in args.inputs:
        with open(p) as f:
            d = json.load(f)
        vals.append(float(d.get("step_accuracy", d.get("overall_step_accuracy", 0.0))))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(args.labels, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Step accuracy")
    ax.set_title("BFS Step Accuracy Comparison")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
