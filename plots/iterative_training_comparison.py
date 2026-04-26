"""
Cross-model pre/post retraining comparison figures.

Input summary JSON is produced by:
  scripts/run_iterative_retraining_pipeline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plots.style import COLOR_BASELINE, COLOR_SFT, apply_paper_style


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        p = out_base.with_suffix(suffix)
        fig.savefig(p, bbox_inches="tight")
        print(f"Wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True, help="Summary JSON from iterative pipeline")
    ap.add_argument("--out-prefix", required=True, help="Figure path prefix")
    args = ap.parse_args()

    with open(args.summary) as f:
        s = json.load(f)
    rows = s.get("models", [])
    if not rows:
        raise RuntimeError("No model rows in summary JSON")

    apply_paper_style()
    import matplotlib.pyplot as plt

    labels = [r["model_label"] for r in rows]
    pre_acc = [float(r["pre"]["step_accuracy"]) for r in rows]
    post_acc = [float(r["post"]["step_accuracy"]) for r in rows]
    pre_fail = [float(r["pre"]["failure_rate"]) for r in rows]
    post_fail = [float(r["post"]["failure_rate"]) for r in rows]

    x = range(len(labels))
    w = 0.36

    # Step accuracy figure
    fig1, ax1 = plt.subplots(figsize=(max(6.5, 1.8 * len(labels)), 4.2))
    ax1.bar([i - w / 2 for i in x], pre_acc, width=w, color=COLOR_BASELINE, label="Pre-retrain")
    ax1.bar([i + w / 2 for i in x], post_acc, width=w, color=COLOR_SFT, label="Post-retrain")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Step accuracy")
    ax1.set_title(f"{s.get('algorithm', 'algorithm').upper()} step accuracy by model (pre vs post)")
    ax1.legend(frameon=False)
    fig1.tight_layout()
    _save(fig1, Path(f"{args.out_prefix}_accuracy"))
    plt.close(fig1)

    # Failure-rate figure (lower is better)
    fig2, ax2 = plt.subplots(figsize=(max(6.5, 1.8 * len(labels)), 4.2))
    ax2.bar([i - w / 2 for i in x], pre_fail, width=w, color=COLOR_BASELINE, label="Pre-retrain")
    ax2.bar([i + w / 2 for i in x], post_fail, width=w, color=COLOR_SFT, label="Post-retrain")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Failure rate")
    ax2.set_title(f"{s.get('algorithm', 'algorithm').upper()} failure rate by model (pre vs post)")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    _save(fig2, Path(f"{args.out_prefix}_failure_rate"))
    plt.close(fig2)


if __name__ == "__main__":
    main()
