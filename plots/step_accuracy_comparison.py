from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _apply_style() -> None:
    p = Path(__file__).resolve().parent / "style.py"
    spec = importlib.util.spec_from_file_location("got_plots_style", p)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_paper_style()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Metric JSON files.")
    ap.add_argument("--labels", nargs="+", required=True, help="Bar labels matching inputs.")
    ap.add_argument("--out", required=True, help="Output PNG or PDF path.")
    args = ap.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    try:
        _apply_style()
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting: pip install matplotlib") from e

    vals = []
    for p in args.inputs:
        with open(p) as f:
            d = json.load(f)
        vals.append(float(d.get("step_accuracy", d.get("overall_step_accuracy", 0.0))))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(args.labels, vals, color="#0072B2")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Step accuracy")
    ax.set_title("BFS step accuracy comparison")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
