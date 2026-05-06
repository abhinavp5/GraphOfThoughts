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


def _load_rates(path: str) -> dict[str, float]:
    with open(path) as f:
        d = json.load(f)
    return {k: float(v) for k, v in d.get("failure_rate_by_gold_operation_type", {}).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Failure-analysis JSON files.")
    ap.add_argument("--labels", nargs="+", required=True, help="Legend labels.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument("--top-k", type=int, default=8, help="Top operation types by avg failure rate.")
    args = ap.parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have same length")

    try:
        _apply_style()
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting: pip install matplotlib") from e

    runs = [_load_rates(p) for p in args.inputs]
    ops = sorted({op for r in runs for op in r.keys()})
    avg = {op: sum(r.get(op, 0.0) for r in runs) / max(1, len(runs)) for op in ops}
    ops = sorted(ops, key=lambda x: avg[x], reverse=True)[: args.top_k]

    width = 0.8 / max(1, len(runs))
    xs = list(range(len(ops)))

    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (label, r) in enumerate(zip(args.labels, runs)):
        offsets = [x - 0.4 + width / 2 + i * width for x in xs]
        vals = [r.get(op, 0.0) for op in ops]
        ax.bar(offsets, vals, width=width, label=label, color=palette[i % len(palette)])

    ax.set_xticks(xs)
    ax.set_xticklabels(ops, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Failure rate")
    ax.set_title("Failure by gold operation type")
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
