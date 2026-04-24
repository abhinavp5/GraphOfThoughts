from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, label in zip(args.inputs, args.labels):
        vals = _extract_first_errors(p)
        if not vals:
            continue
        ax.hist(vals, bins=20, alpha=0.4, label=label)

    ax.set_xlabel("First error step")
    ax.set_ylabel("Count")
    ax.set_title("First-Error Step Distribution")
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
