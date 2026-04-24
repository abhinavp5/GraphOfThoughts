"""
Build compact BFS final comparison table from metric JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="Run labels (baseline/sft/dagger).")
    ap.add_argument("--main-metrics", nargs="+", required=True, help="Main operation_accuracy json files.")
    ap.add_argument("--main-failures", nargs="+", required=True, help="Main failure_analysis json files.")
    ap.add_argument("--nlgraph-metrics", nargs="+", required=True, help="NLGraph operation_accuracy json files.")
    ap.add_argument("--glbench-metrics", nargs="+", required=True, help="GLBench operation_accuracy json files.")
    ap.add_argument("--out", required=True, help="Output markdown table path.")
    args = ap.parse_args()

    n = len(args.labels)
    fields = [args.main_metrics, args.main_failures, args.nlgraph_metrics, args.glbench_metrics]
    if any(len(x) != n for x in fields):
        raise ValueError("All metric/failure lists must match number of labels")

    lines = [
        "| Model | Main Step Acc | Main Mean First Error | Main Failure Rate | NLGraph Step Acc | GLBench Step Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, label in enumerate(args.labels):
        mm = _read(args.main_metrics[i])
        mf = _read(args.main_failures[i])
        nl = _read(args.nlgraph_metrics[i])
        gl = _read(args.glbench_metrics[i])
        lines.append(
            "| "
            f"{label} | "
            f"{float(mm.get('step_accuracy', 0.0)):.4f} | "
            f"{float(mm.get('mean_first_error_step') or 0.0):.2f} | "
            f"{float(mf.get('overall_failure_rate', 0.0)):.4f} | "
            f"{float(nl.get('step_accuracy', nl.get('overall_step_accuracy', 0.0))):.4f} | "
            f"{float(gl.get('step_accuracy', gl.get('overall_step_accuracy', 0.0))):.4f} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
