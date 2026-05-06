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


def _extract_first_errors(path: str) -> list[int]:
    with open(path) as f:
        d = json.load(f)
    rows = d.get("per_sample", [])
    out = []
    for r in rows:
        x = r.get("first_error_step")
        if x is not None:
            out.append(int(x))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Failure-analysis JSON files.")
    ap.add_argument("--labels", nargs="+", required=True, help="Legend labels.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    args = ap.parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have same length")

    try:
        _apply_style()
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    all_hi = 1
    for label, path, c in zip(args.labels, args.inputs, colors):
        vals = _extract_first_errors(path)
        if not vals:
            continue
        all_hi = max(all_hi, max(vals) + 1)
        bins = np.arange(-0.5, max(vals) + 1.5, 1.0)
        ax.hist(vals, bins=bins, alpha=0.5, label=label, color=c, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("First error step")
    ax.set_ylabel("Count")
    ax.set_title("First-error step distribution")
    ax.set_xlim(-0.5, all_hi - 0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
