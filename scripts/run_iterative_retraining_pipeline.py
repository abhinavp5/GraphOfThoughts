"""
Iterative train -> evaluate -> error-trace retrain pipeline across models.

For each provided training config:
  1) Train LoRA adapter with training.sft
  2) Evaluate pre-retrain model via scripts/run_pipeline.sh
  3) Collect error traces via training.dagger collect
  4) Retrain on recovery traces via training.dagger finetune
  5) Re-evaluate post-retrain model via scripts/run_pipeline.sh
  6) Save per-model comparison JSON

Finally writes a cross-model summary JSON and paper figures comparing:
  - pre vs post step accuracy
  - pre vs post failure rate

All outputs are timestamped under one root directory so nothing is overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-")


def _tag(algorithm: str, family: str, n: int, count: int, seed: int) -> str:
    return f"{algorithm}_{family}_n{n}_c{count}_s{seed}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _prepare_config_copy(
    *,
    src_config: Path,
    dst_config: Path,
    output_dir: Path,
    train_epochs_override: int | None,
) -> dict[str, Any]:
    with src_config.open() as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("training", {})
    cfg["training"]["output_dir"] = str(output_dir)
    if train_epochs_override is not None:
        cfg["training"]["epochs"] = train_epochs_override
    dst_config.parent.mkdir(parents=True, exist_ok=True)
    with dst_config.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "training/configs/qwen_2_5_7b.yaml",
            "training/configs/llama_3_1b.yaml",
        ],
        help="Training config YAMLs (one model per config).",
    )
    ap.add_argument("--algorithm", default="bfs", choices=["bfs", "dfs", "dijkstra"])
    ap.add_argument("--family", default="erdos_renyi")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--bench-limit", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--skip-smoke", action="store_true", default=True)
    ap.add_argument("--skip-data-after-first", action="store_true", default=True)
    ap.add_argument("--nlgraph-input", default="data/benchmarks/nlgraph_eval.json")
    ap.add_argument("--glbench-input", default="data/benchmarks/glbench_eval.json")
    ap.add_argument("--train-epochs-override", type=int, default=None)
    ap.add_argument("--dagger-epochs", type=int, default=1)
    ap.add_argument("--dagger-batch-size", type=int, default=2)
    ap.add_argument("--dagger-grad-accum", type=int, default=4)
    ap.add_argument("--dagger-lr", type=float, default=1e-5)
    ap.add_argument("--train-max-n", type=int, default=10)
    ap.add_argument("--out-root", default="out/iterative_runs")
    ap.add_argument("--paper-fig-dir", default="paper/figures")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = (repo_root / args.out_root / f"{args.algorithm}_iterative_{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    tag = _tag(args.algorithm, args.family, args.n, args.count, args.seed)

    print(f"[info] run root: {run_root}")
    print(f"[info] run tag:  {tag}")

    model_rows: list[dict[str, Any]] = []
    generated_data_once = False

    for idx, cfg_rel in enumerate(args.configs):
        cfg_path = (repo_root / cfg_rel).resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Training config not found: {cfg_path}")

        model_label = _slug(cfg_path.stem)
        model_root = run_root / model_label
        pre_eval_dir = model_root / "eval_pre"
        post_eval_dir = model_root / "eval_post"
        pre_adapter_dir = model_root / "pre_sft_adapter"
        dagger_dir = model_root / "dagger"
        post_model_dir = model_root / "post_retrain_model"
        config_copy = model_root / "resolved_train_config.yaml"
        recovery_json = dagger_dir / f"recovery_traces_{tag}.json"

        model_root.mkdir(parents=True, exist_ok=True)

        cfg = _prepare_config_copy(
            src_config=cfg_path,
            dst_config=config_copy,
            output_dir=pre_adapter_dir,
            train_epochs_override=args.train_epochs_override,
        )
        base_model = str(cfg["model"]["name"])
        print(f"\n[model] {model_label} ({base_model})")

        # 1) Initial SFT training
        _run(
            ["python", "-m", "training.sft", "--config", str(config_copy)],
            cwd=repo_root,
        )

        # 2) Pre-retrain evaluation
        cmd_pre = [
            "bash",
            "scripts/run_pipeline.sh",
            "--model",
            base_model,
            "--adapter",
            str(pre_adapter_dir),
            "--algorithm",
            args.algorithm,
            "--family",
            args.family,
            "--n",
            str(args.n),
            "--count",
            str(args.count),
            "--seed",
            str(args.seed),
            "--limit",
            str(args.limit),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--out-dir",
            str(pre_eval_dir),
            "--nlgraph-input",
            args.nlgraph_input,
            "--glbench-input",
            args.glbench_input,
            "--bench-limit",
            str(args.bench_limit),
        ]
        if args.skip_smoke:
            cmd_pre.append("--skip-smoke")
        if args.skip_data_after_first and generated_data_once:
            cmd_pre.append("--skip-data")
        _run(cmd_pre, cwd=repo_root)
        generated_data_once = True

        # 3) Collect error traces for retraining
        _run(
            [
                "python",
                "-m",
                "training.dagger",
                "collect",
                "--model",
                base_model,
                "--adapter",
                str(pre_adapter_dir),
                "--trace-dir",
                "data/traces",
                "--trace-pattern",
                f"pipe_{tag}.json",
                "--out",
                str(recovery_json),
                "--limit",
                str(args.count),
                "--device",
                args.device,
                "--dtype",
                "float16" if args.dtype == "auto" else args.dtype,
                "--seed",
                str(args.seed),
            ],
            cwd=repo_root,
        )

        # 4) Retrain from collected error traces
        _run(
            [
                "python",
                "-m",
                "training.dagger",
                "finetune",
                "--model",
                base_model,
                "--recovery-json",
                str(recovery_json),
                "--output-dir",
                str(post_model_dir),
                "--epochs",
                str(args.dagger_epochs),
                "--batch-size",
                str(args.dagger_batch_size),
                "--grad-accum",
                str(args.dagger_grad_accum),
                "--lr",
                str(args.dagger_lr),
            ],
            cwd=repo_root,
        )

        # 5) Post-retrain evaluation (use post model directory directly)
        cmd_post = [
            "bash",
            "scripts/run_pipeline.sh",
            "--model",
            str(post_model_dir),
            "--algorithm",
            args.algorithm,
            "--family",
            args.family,
            "--n",
            str(args.n),
            "--count",
            str(args.count),
            "--seed",
            str(args.seed),
            "--limit",
            str(args.limit),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--out-dir",
            str(post_eval_dir),
            "--nlgraph-input",
            args.nlgraph_input,
            "--glbench-input",
            args.glbench_input,
            "--bench-limit",
            str(args.bench_limit),
            "--skip-data",
        ]
        if args.skip_smoke:
            cmd_post.append("--skip-smoke")
        _run(cmd_post, cwd=repo_root)

        # Optional extra reports for pre/post
        for phase, eval_dir in (("pre", pre_eval_dir), ("post", post_eval_dir)):
            pred = eval_dir / f"pred_{tag}.json"
            _run(
                [
                    "python",
                    "-m",
                    "evaluation.metrics.state_consistency",
                    str(pred),
                    "--out",
                    str(eval_dir / f"state_consistency_{tag}.json"),
                ],
                cwd=repo_root,
            )
            _run(
                [
                    "python",
                    "-m",
                    "evaluation.metrics.structural_generalization",
                    str(pred),
                    "--train-max-n",
                    str(args.train_max_n),
                    "--out",
                    str(eval_dir / f"structural_generalization_{tag}.json"),
                ],
                cwd=repo_root,
            )
            print(f"[info] wrote {phase}-retrain consistency + structural reports")

        # 6) Per-model comparison summary
        pre_metrics = _load_json(pre_eval_dir / f"metrics_{tag}.json")
        post_metrics = _load_json(post_eval_dir / f"metrics_{tag}.json")
        pre_failures = _load_json(pre_eval_dir / f"failures_{tag}.json")
        post_failures = _load_json(post_eval_dir / f"failures_{tag}.json")
        pre_nl = _load_json(pre_eval_dir / f"nlgraph_{tag}_operation_accuracy.json")
        post_nl = _load_json(post_eval_dir / f"nlgraph_{tag}_operation_accuracy.json")
        pre_gl = _load_json(pre_eval_dir / f"glbench_{tag}_operation_accuracy.json")
        post_gl = _load_json(post_eval_dir / f"glbench_{tag}_operation_accuracy.json")

        row = {
            "model_label": model_label,
            "base_model": base_model,
            "paths": {
                "model_root": str(model_root),
                "pre_adapter_dir": str(pre_adapter_dir),
                "recovery_json": str(recovery_json),
                "post_model_dir": str(post_model_dir),
                "pre_eval_dir": str(pre_eval_dir),
                "post_eval_dir": str(post_eval_dir),
            },
            "pre": {
                "step_accuracy": float(pre_metrics.get("step_accuracy", 0.0)),
                "failure_rate": float(pre_failures.get("overall_failure_rate", 0.0)),
                "mean_first_error_step": pre_metrics.get("mean_first_error_step"),
                "nlgraph_step_accuracy": float(pre_nl.get("step_accuracy", 0.0)),
                "glbench_step_accuracy": float(pre_gl.get("step_accuracy", 0.0)),
            },
            "post": {
                "step_accuracy": float(post_metrics.get("step_accuracy", 0.0)),
                "failure_rate": float(post_failures.get("overall_failure_rate", 0.0)),
                "mean_first_error_step": post_metrics.get("mean_first_error_step"),
                "nlgraph_step_accuracy": float(post_nl.get("step_accuracy", 0.0)),
                "glbench_step_accuracy": float(post_gl.get("step_accuracy", 0.0)),
            },
        }
        row["delta"] = {
            "step_accuracy": row["post"]["step_accuracy"] - row["pre"]["step_accuracy"],
            "failure_rate": row["post"]["failure_rate"] - row["pre"]["failure_rate"],
            "mean_first_error_step": (
                (row["post"]["mean_first_error_step"] or 0.0)
                - (row["pre"]["mean_first_error_step"] or 0.0)
            ),
            "nlgraph_step_accuracy": row["post"]["nlgraph_step_accuracy"] - row["pre"]["nlgraph_step_accuracy"],
            "glbench_step_accuracy": row["post"]["glbench_step_accuracy"] - row["pre"]["glbench_step_accuracy"],
        }
        _write_json(model_root / "comparison_pre_vs_post.json", row)
        model_rows.append(row)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "algorithm": args.algorithm,
        "family": args.family,
        "n": args.n,
        "count": args.count,
        "seed": args.seed,
        "limit": args.limit,
        "bench_limit": args.bench_limit,
        "run_root": str(run_root),
        "models": model_rows,
    }
    summary_path = run_root / "summary_iterative_pre_post.json"
    _write_json(summary_path, summary)
    print(f"\n[done] summary: {summary_path}")

    # Paper figures for cross-model pre/post comparisons
    fig_dir = (repo_root / args.paper_fig_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "python",
            "-m",
            "plots.iterative_training_comparison",
            "--summary",
            str(summary_path),
            "--out-prefix",
            str(fig_dir / f"fig_iterative_{_slug(args.algorithm)}"),
        ],
        cwd=repo_root,
    )

    # Keep an easy-to-find latest pointer
    latest_dir = (repo_root / args.out_root / "latest").resolve()
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    try:
        latest_dir.symlink_to(run_root, target_is_directory=True)
    except OSError:
        shutil.copytree(run_root, latest_dir)
    print(f"[done] latest run pointer: {latest_dir}")


if __name__ == "__main__":
    main()
