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
    ap.add_argument("--report", required=True, help="JSON from evaluation.metrics.structural_generalization")
    ap.add_argument("--out", required=True, help="Output PNG path")
    args = ap.parse_args()

    try:
        _apply_style()
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting: pip install matplotlib") from e

    with open(args.report) as f:
        d = json.load(f)
    by_n = d.get("by_graph_size", {})
    xs = sorted(int(k) for k in by_n.keys())
    train_max_n = int(d.get("train_max_n", 0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(xs) >= 2:
        ys = [float(by_n[str(x)]["step_accuracy"]) for x in xs]
        ax.plot(xs, ys, marker="o", color="#D55E00", linewidth=2)
        if train_max_n > 0:
            ax.axvline(train_max_n, linestyle="--", color="#666666", label=f"Train max n = {train_max_n}")
        ax.legend(frameon=False)
    else:
        id_acc = float(d.get("in_distribution_step_accuracy", 0.0))
        ax.bar([0], [id_acc], color="#D55E00", width=0.5)
        ax.set_xticks([0])
        n_label = str(xs[0]) if xs else "?"
        ax.set_xticklabels([f"n = {n_label} (eval)"])
        ax.text(
            0.02,
            0.02,
            f"Train max n = {train_max_n}. Plot a multi-n eval for a curve.",
            transform=ax.transAxes,
            fontsize=8,
            color="#444444",
        )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Graph size (n)" if len(xs) >= 2 else "")
    ax.set_ylabel("Step accuracy")
    ax.set_title("Structural generalization (BFS)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
