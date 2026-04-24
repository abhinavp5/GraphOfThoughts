"""
Log a pipeline run manifest to Weights & Biases (aggregate metrics only).

Reads run_manifest_*.json from run_pipeline.sh and logs scalar summary metrics
for pipe / nlgraph / glbench where corresponding metric files exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _flatten_metrics(d: dict[str, Any]) -> dict[str, int | float | bool]:
    """Top-level numeric/bool metrics from operation_accuracy JSON (skip nested)."""
    out: dict[str, int | float | bool] = {}
    skip = frozenset({"per_sample"})
    for k, v in d.items():
        if k in skip:
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int) and not isinstance(v, bool):
            out[k] = v
        elif isinstance(v, float):
            if v == v:  # not NaN
                out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="W&B log for pipeline run manifest")
    ap.add_argument(
        "--manifest",
        required=True,
        help="Path to run_manifest_*.json from run_pipeline.sh",
    )
    args = ap.parse_args()

    project = os.environ.get("WANDB_PROJECT")
    if not project:
        print("WANDB_PROJECT not set; skip W&B", file=sys.stderr)
        sys.exit(0)

    with open(args.manifest) as f:
        manifest: dict = json.load(f)

    tag = manifest.get("tag", "run")
    run_id = manifest.get("run_id", "")
    name = f"{tag}_{run_id}"[:128]

    import wandb

    cfg = dict(manifest.get("config", {}))
    paths = manifest.get("paths", {})
    if paths.get("log_file"):
        cfg["log_file"] = paths["log_file"]

    to_log: dict[str, int | float | bool] = {}

    pipe_m = paths.get("pipe", {})
    if isinstance(pipe_m, dict) and pipe_m.get("operation_accuracy"):
        p = pipe_m["operation_accuracy"]
        if p and os.path.isfile(p):
            with open(p) as f:
                data = json.load(f)
            for k, v in _flatten_metrics(data).items():
                to_log[f"pipe/{k}"] = v

    for bench, key in (("nlgraph", "nlgraph"), ("glbench", "glbench")):
        b = paths.get(bench)
        if not isinstance(b, dict):
            continue
        apath = b.get("operation_accuracy")
        if not apath or not os.path.isfile(apath):
            continue
        with open(apath) as f:
            data = json.load(f)
        for k, v in _flatten_metrics(data).items():
            to_log[f"{key}/{k}"] = v

    try:
        wandb.init(project=project, name=name, config=cfg)
        if to_log:
            wandb.log(to_log)
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
