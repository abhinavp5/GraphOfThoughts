from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="JSON from evaluation.metrics.structural_generalization")
    ap.add_argument("--out", required=True, help="Output PNG path")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting: pip install matplotlib") from e

    with open(args.report) as f:
        d = json.load(f)
    by_n = d.get("by_graph_size", {})
    xs = sorted(int(k) for k in by_n.keys())
    ys = [float(by_n[str(x)]["step_accuracy"]) for x in xs]
    train_max_n = int(d.get("train_max_n", 0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o")
    if train_max_n > 0:
        ax.axvline(train_max_n, linestyle="--", color="red", label=f"train max n={train_max_n}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Graph size (n)")
    ax.set_ylabel("Step accuracy")
    ax.set_title("Structural Generalization (BFS)")
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
