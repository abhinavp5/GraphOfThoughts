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


_BASE_ALGOS: list[str] = ["bfs", "dfs", "dijkstra"]


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


def _load_json_list(path: Path) -> list[Any]:
    with path.open() as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    return [obj]


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
    ap.add_argument(
        "--algorithm",
        default="bfs",
        choices=[*_BASE_ALGOS, "all", "mixed"],
        help="bfs|dfs|dijkstra (single), all (run all 3 separately), mixed (merge recovery data across all 3 then eval all 3).",
    )
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


def _run_pipeline_eval(
    *,
    repo_root: Path,
    model: str,
    adapter: str | None,
    algorithm: str,
    family: str,
    n: int,
    count: int,
    seed: int,
    limit: int,
    bench_limit: int,
    device: str,
    dtype: str,
    out_dir: Path,
    nlgraph_input: str,
    glbench_input: str,
    skip_smoke: bool,
    skip_data: bool,
) -> None:
    cmd = [
        "bash",
        "scripts/run_pipeline.sh",
        "--model",
        model,
        "--algorithm",
        algorithm,
        "--family",
        family,
        "--n",
        str(n),
        "--count",
        str(count),
        "--seed",
        str(seed),
        "--limit",
        str(limit),
        "--device",
        device,
        "--dtype",
        dtype,
        "--out-dir",
        str(out_dir),
        "--nlgraph-input",
        nlgraph_input,
        "--glbench-input",
        glbench_input,
        "--bench-limit",
        str(bench_limit),
    ]
    if adapter:
        cmd.extend(["--adapter", adapter])
    if skip_smoke:
        cmd.append("--skip-smoke")
    if skip_data:
        cmd.append("--skip-data")
    _run(cmd, cwd=repo_root)


def _run_consistency_reports(
    *,
    repo_root: Path,
    eval_dir: Path,
    tag: str,
    train_max_n: int,
) -> None:
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
            str(train_max_n),
            "--out",
            str(eval_dir / f"structural_generalization_{tag}.json"),
        ],
        cwd=repo_root,
    )


def _row_from_eval_dirs(
    *,
    model_label: str,
    base_model: str,
    model_root: Path,
    pre_adapter_dir: Path | None,
    recovery_json: Path | None,
    post_model_dir: Path | None,
    pre_eval_dir: Path,
    post_eval_dir: Path,
    tag: str,
) -> dict[str, Any]:
    pre_metrics = _load_json(pre_eval_dir / f"metrics_{tag}.json")
    post_metrics = _load_json(post_eval_dir / f"metrics_{tag}.json")
    pre_failures = _load_json(pre_eval_dir / f"failures_{tag}.json")
    post_failures = _load_json(post_eval_dir / f"failures_{tag}.json")
    pre_nl = _load_json(pre_eval_dir / f"nlgraph_{tag}_operation_accuracy.json")
    post_nl = _load_json(post_eval_dir / f"nlgraph_{tag}_operation_accuracy.json")
    pre_gl = _load_json(pre_eval_dir / f"glbench_{tag}_operation_accuracy.json")
    post_gl = _load_json(post_eval_dir / f"glbench_{tag}_operation_accuracy.json")

    row: dict[str, Any] = {
        "model_label": model_label,
        "base_model": base_model,
        "paths": {
            "model_root": str(model_root),
            "pre_adapter_dir": str(pre_adapter_dir) if pre_adapter_dir else None,
            "recovery_json": str(recovery_json) if recovery_json else None,
            "post_model_dir": str(post_model_dir) if post_model_dir else None,
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
    return row


def _run_single_algorithm(args: argparse.Namespace, *, repo_root: Path, algorithm: str, stamp: str) -> None:
    run_root = (repo_root / args.out_root / f"{algorithm}_iterative_{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    tag = _tag(algorithm, args.family, args.n, args.count, args.seed)

    print(f"[info] run root: {run_root}")
    print(f"[info] run tag:  {tag}")

    model_rows: list[dict[str, Any]] = []
    generated_data_once = False

    for cfg_rel in args.configs:
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
        _run_pipeline_eval(
            repo_root=repo_root,
            model=base_model,
            adapter=str(pre_adapter_dir),
            algorithm=algorithm,
            family=args.family,
            n=args.n,
            count=args.count,
            seed=args.seed,
            limit=args.limit,
            bench_limit=args.bench_limit,
            device=args.device,
            dtype=args.dtype,
            out_dir=pre_eval_dir,
            nlgraph_input=args.nlgraph_input,
            glbench_input=args.glbench_input,
            skip_smoke=bool(args.skip_smoke),
            skip_data=bool(args.skip_data_after_first and generated_data_once),
        )
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
        recovery_records: list[Any] = []
        try:
            recovery_records = _load_json_list(recovery_json)
        except FileNotFoundError:
            recovery_records = []

        retrained_model_dir: Path | None = None
        retrain_skipped_reason: str | None = None
        if len(recovery_records) == 0:
            retrain_skipped_reason = "no_recovery_examples"
            print(f"[warn] {model_label} / {algorithm}: 0 recovery examples; skipping retrain + using pre adapter for post-eval")
        else:
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
            retrained_model_dir = post_model_dir

        # 5) Post-retrain evaluation
        # - If retrain ran: evaluate the saved model directory directly.
        # - If retrain was skipped: evaluate the same (base_model + pre_adapter_dir) again to keep the pipeline moving.
        if retrained_model_dir is not None:
            _run_pipeline_eval(
                repo_root=repo_root,
                model=str(retrained_model_dir),
                adapter=None,
                algorithm=algorithm,
                family=args.family,
                n=args.n,
                count=args.count,
                seed=args.seed,
                limit=args.limit,
                bench_limit=args.bench_limit,
                device=args.device,
                dtype=args.dtype,
                out_dir=post_eval_dir,
                nlgraph_input=args.nlgraph_input,
                glbench_input=args.glbench_input,
                skip_smoke=bool(args.skip_smoke),
                skip_data=True,
            )
        else:
            _run_pipeline_eval(
                repo_root=repo_root,
                model=base_model,
                adapter=str(pre_adapter_dir),
                algorithm=algorithm,
                family=args.family,
                n=args.n,
                count=args.count,
                seed=args.seed,
                limit=args.limit,
                bench_limit=args.bench_limit,
                device=args.device,
                dtype=args.dtype,
                out_dir=post_eval_dir,
                nlgraph_input=args.nlgraph_input,
                glbench_input=args.glbench_input,
                skip_smoke=bool(args.skip_smoke),
                skip_data=True,
            )

        # Optional extra reports for pre/post
        for phase, eval_dir in (("pre", pre_eval_dir), ("post", post_eval_dir)):
            _run_consistency_reports(
                repo_root=repo_root,
                eval_dir=eval_dir,
                tag=tag,
                train_max_n=args.train_max_n,
            )
            print(f"[info] wrote {phase}-retrain consistency + structural reports")

        # 6) Per-model comparison summary
        row = _row_from_eval_dirs(
            model_label=model_label,
            base_model=base_model,
            model_root=model_root,
            pre_adapter_dir=pre_adapter_dir,
            recovery_json=recovery_json,
            post_model_dir=retrained_model_dir,
            pre_eval_dir=pre_eval_dir,
            post_eval_dir=post_eval_dir,
            tag=tag,
        )
        if retrain_skipped_reason is not None:
            row["retrain_skipped_reason"] = retrain_skipped_reason
        _write_json(model_root / "comparison_pre_vs_post.json", row)
        model_rows.append(row)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "algorithm": algorithm,
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
            str(fig_dir / f"fig_iterative_{_slug(algorithm)}"),
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


def _run_mixed(args: argparse.Namespace, *, repo_root: Path, stamp: str) -> None:
    run_root = (repo_root / args.out_root / f"mixed_iterative_{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] run root: {run_root}")
    print(f"[info] mode:     mixed (merge recovery examples across {','.join(_BASE_ALGOS)})")

    # Track per-algo rows so we can write per-algo summaries compatible with the existing plotter.
    model_rows_by_algo: dict[str, list[dict[str, Any]]] = {a: [] for a in _BASE_ALGOS}
    generated_data_once_by_algo: dict[str, bool] = {a: False for a in _BASE_ALGOS}

    for cfg_rel in args.configs:
        cfg_path = (repo_root / cfg_rel).resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Training config not found: {cfg_path}")

        model_label = _slug(cfg_path.stem)
        model_root = run_root / model_label
        pre_adapter_dir = model_root / "pre_sft_adapter"
        dagger_dir = model_root / "dagger"
        post_model_dir = model_root / "post_retrain_model"
        config_copy = model_root / "resolved_train_config.yaml"
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

        # 2) Pre-retrain evaluation + 3) collect recovery examples for each algo
        merged_records: list[dict[str, Any]] = []
        recovery_jsons: dict[str, Path] = {}

        for algo in _BASE_ALGOS:
            tag = _tag(algo, args.family, args.n, args.count, args.seed)
            pre_eval_dir = model_root / f"eval_pre_{algo}"
            recovery_json = dagger_dir / f"recovery_traces_{tag}.json"
            recovery_jsons[algo] = recovery_json

            _run_pipeline_eval(
                repo_root=repo_root,
                model=base_model,
                adapter=str(pre_adapter_dir),
                algorithm=algo,
                family=args.family,
                n=args.n,
                count=args.count,
                seed=args.seed,
                limit=args.limit,
                bench_limit=args.bench_limit,
                device=args.device,
                dtype=args.dtype,
                out_dir=pre_eval_dir,
                nlgraph_input=args.nlgraph_input,
                glbench_input=args.glbench_input,
                skip_smoke=bool(args.skip_smoke),
                skip_data=bool(args.skip_data_after_first and generated_data_once_by_algo[algo]),
            )
            generated_data_once_by_algo[algo] = True

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

            try:
                with recovery_json.open() as f:
                    merged_records.extend(json.load(f))
            except FileNotFoundError:
                # If collect produced nothing, keep going; finetune will error later if the full merge is empty.
                pass

            _run_consistency_reports(
                repo_root=repo_root,
                eval_dir=pre_eval_dir,
                tag=tag,
                train_max_n=args.train_max_n,
            )

        # Merge recovery JSONs
        merged_recovery_json = dagger_dir / f"recovery_traces_mixed_{_slug(args.family)}_n{args.n}_c{args.count}_s{args.seed}.json"
        merged_recovery_json.parent.mkdir(parents=True, exist_ok=True)
        with merged_recovery_json.open("w") as f:
            json.dump(merged_records, f, indent=2)
            f.write("\n")
        print(f"[info] merged recovery examples: {len(merged_records)} -> {merged_recovery_json}")

        # 4) Finetune once on merged recovery examples (skip if empty)
        merged_retrain_skipped_reason: str | None = None
        if len(merged_records) == 0:
            merged_retrain_skipped_reason = "no_recovery_examples"
            print(f"[warn] {model_label} / mixed: 0 merged recovery examples; skipping retrain + using pre adapter for post-eval")
            retrained_model_dir: Path | None = None
        else:
            _run(
                [
                    "python",
                    "-m",
                    "training.dagger",
                    "finetune",
                    "--model",
                    base_model,
                    "--recovery-json",
                    str(merged_recovery_json),
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
            retrained_model_dir = post_model_dir

        # 5) Post-retrain evaluation on ALL 3 algorithms
        for algo in _BASE_ALGOS:
            tag = _tag(algo, args.family, args.n, args.count, args.seed)
            post_eval_dir = model_root / f"eval_post_{algo}"
            if retrained_model_dir is not None:
                _run_pipeline_eval(
                    repo_root=repo_root,
                    model=str(retrained_model_dir),
                    adapter=None,
                    algorithm=algo,
                    family=args.family,
                    n=args.n,
                    count=args.count,
                    seed=args.seed,
                    limit=args.limit,
                    bench_limit=args.bench_limit,
                    device=args.device,
                    dtype=args.dtype,
                    out_dir=post_eval_dir,
                    nlgraph_input=args.nlgraph_input,
                    glbench_input=args.glbench_input,
                    skip_smoke=bool(args.skip_smoke),
                    skip_data=True,
                )
            else:
                _run_pipeline_eval(
                    repo_root=repo_root,
                    model=base_model,
                    adapter=str(pre_adapter_dir),
                    algorithm=algo,
                    family=args.family,
                    n=args.n,
                    count=args.count,
                    seed=args.seed,
                    limit=args.limit,
                    bench_limit=args.bench_limit,
                    device=args.device,
                    dtype=args.dtype,
                    out_dir=post_eval_dir,
                    nlgraph_input=args.nlgraph_input,
                    glbench_input=args.glbench_input,
                    skip_smoke=bool(args.skip_smoke),
                    skip_data=True,
                )

            _run_consistency_reports(
                repo_root=repo_root,
                eval_dir=post_eval_dir,
                tag=tag,
                train_max_n=args.train_max_n,
            )

            # Per-algo row (compatible with existing plotter via per-algo summaries)
            row = _row_from_eval_dirs(
                model_label=model_label,
                base_model=base_model,
                model_root=model_root,
                pre_adapter_dir=pre_adapter_dir,
                recovery_json=merged_recovery_json,
                post_model_dir=retrained_model_dir,
                pre_eval_dir=(model_root / f"eval_pre_{algo}"),
                post_eval_dir=post_eval_dir,
                tag=tag,
            )
            row["mode"] = "mixed"
            row["algorithm"] = algo
            if merged_retrain_skipped_reason is not None:
                row["retrain_skipped_reason"] = merged_retrain_skipped_reason
            row["paths"]["merged_recovery_json"] = str(merged_recovery_json)
            row["paths"]["recovery_jsons"] = {a: str(p) for a, p in recovery_jsons.items()}
            _write_json(model_root / f"comparison_pre_vs_post_{algo}.json", row)
            model_rows_by_algo[algo].append(row)

    # Write per-algo summaries so the existing plotter can be reused unchanged.
    fig_dir = (repo_root / args.paper_fig_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    for algo in _BASE_ALGOS:
        tag = _tag(algo, args.family, args.n, args.count, args.seed)
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tag": tag,
            "algorithm": algo,
            "family": args.family,
            "n": args.n,
            "count": args.count,
            "seed": args.seed,
            "limit": args.limit,
            "bench_limit": args.bench_limit,
            "run_root": str(run_root),
            "mode": "mixed",
            "models": model_rows_by_algo[algo],
        }
        summary_path = run_root / f"summary_iterative_pre_post_{algo}.json"
        _write_json(summary_path, summary)
        print(f"[done] mixed summary ({algo}): {summary_path}")
        _run(
            [
                "python",
                "-m",
                "plots.iterative_training_comparison",
                "--summary",
                str(summary_path),
                "--out-prefix",
                str(fig_dir / f"fig_iterative_mixed_{_slug(algo)}"),
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


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.algorithm == "all":
        for algo in _BASE_ALGOS:
            _run_single_algorithm(args, repo_root=repo_root, algorithm=algo, stamp=stamp)
        return

    if args.algorithm == "mixed":
        _run_mixed(args, repo_root=repo_root, stamp=stamp)
        return

    _run_single_algorithm(args, repo_root=repo_root, algorithm=args.algorithm, stamp=stamp)


if __name__ == "__main__":
    main()
