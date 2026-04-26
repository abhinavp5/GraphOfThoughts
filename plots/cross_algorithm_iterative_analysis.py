"""
Cross-algorithm (BFS vs DFS) pre/post-DAgger analysis figures.

This script aggregates iterative-run outputs and writes a comprehensive
set of comparison figures for paper/report use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plots.style import COLOR_BASELINE, COLOR_SFT, apply_paper_style


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        p = out_base.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight")
        print(f"Wrote {p}")


def _extract_metric(summary: dict[str, Any], key: str) -> tuple[float, float]:
    row = summary["models"][0]
    return float(row["pre"][key]), float(row["post"][key])


def _derive_failure_path(summary: dict[str, Any], phase: str, fallback_run_root: Path) -> Path:
    tag = summary["tag"]
    root = Path(summary.get("run_root", fallback_run_root))
    if not root.exists():
        root = fallback_run_root
    model_label = summary["models"][0]["model_label"]
    return root / model_label / f"eval_{phase}" / f"failures_{tag}.json"


def _phase_rates(path: Path) -> list[float]:
    d = _load(path)
    fr = d.get("failure_rate_by_step_bucket", {})
    return [float(fr.get(k, 0.0)) for k in ("early", "middle", "late")]


def _op_rates(path: Path) -> dict[str, float]:
    d = _load(path)
    return {k: float(v) for k, v in d.get("failure_rate_by_gold_operation_type", {}).items()}


def _sum_top_confusions(path: Path, top_k: int = 10) -> int:
    d = _load(path)
    return int(sum(int(row.get("count", 0)) for row in d.get("top_operation_confusions", [])[:top_k]))


def fig_main_pre_post(
    bfs_summary: dict[str, Any],
    dfs_summary: dict[str, Any],
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = ["BFS", "DFS"]
    pre_step = [_extract_metric(bfs_summary, "step_accuracy")[0], _extract_metric(dfs_summary, "step_accuracy")[0]]
    post_step = [_extract_metric(bfs_summary, "step_accuracy")[1], _extract_metric(dfs_summary, "step_accuracy")[1]]
    pre_fail = [_extract_metric(bfs_summary, "failure_rate")[0], _extract_metric(dfs_summary, "failure_rate")[0]]
    post_fail = [_extract_metric(bfs_summary, "failure_rate")[1], _extract_metric(dfs_summary, "failure_rate")[1]]

    x = list(range(len(labels)))
    w = 0.34
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax1.bar([i - w / 2 for i in x], pre_step, width=w, color=COLOR_BASELINE, label="Pre-DAgger")
    ax1.bar([i + w / 2 for i in x], post_step, width=w, color=COLOR_SFT, label="Post-DAgger")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Step accuracy")
    ax1.set_title("Main-task accuracy")
    ax1.legend(frameon=False)

    ax2.bar([i - w / 2 for i in x], pre_fail, width=w, color=COLOR_BASELINE, label="Pre-DAgger")
    ax2.bar([i + w / 2 for i in x], post_fail, width=w, color=COLOR_SFT, label="Post-DAgger")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Failure rate")
    ax2.set_title("Main-task failure rate (lower is better)")

    fig.suptitle("BFS vs DFS: pre/post DAgger core metrics", y=1.02)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def fig_delta_bars(
    bfs_summary: dict[str, Any],
    dfs_summary: dict[str, Any],
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt

    row_b = bfs_summary["models"][0]
    row_d = dfs_summary["models"][0]

    metrics = [
        ("Main step acc delta", row_b["delta"]["step_accuracy"], row_d["delta"]["step_accuracy"]),
        ("Main fail rate delta", row_b["delta"]["failure_rate"], row_d["delta"]["failure_rate"]),
        ("NLGraph acc delta", row_b["delta"]["nlgraph_step_accuracy"], row_d["delta"]["nlgraph_step_accuracy"]),
        ("GLBench acc delta", row_b["delta"]["glbench_step_accuracy"], row_d["delta"]["glbench_step_accuracy"]),
    ]
    x = list(range(len(metrics)))
    w = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bfs_vals = [float(m[1]) for m in metrics]
    dfs_vals = [float(m[2]) for m in metrics]
    ax.bar([i - w / 2 for i in x], bfs_vals, width=w, color="#1F77B4", label="BFS")
    ax.bar([i + w / 2 for i in x], dfs_vals, width=w, color="#FF7F0E", label="DFS")
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], rotation=18, ha="right")
    ax.set_ylabel("Post - Pre")
    ax.set_title("DAgger gain/loss deltas by algorithm")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def fig_failure_phase_grid(
    bfs_summary: dict[str, Any],
    dfs_summary: dict[str, Any],
    bfs_root: Path,
    dfs_root: Path,
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt

    bfs_pre = _phase_rates(_derive_failure_path(bfs_summary, "pre", bfs_root))
    bfs_post = _phase_rates(_derive_failure_path(bfs_summary, "post", bfs_root))
    dfs_pre = _phase_rates(_derive_failure_path(dfs_summary, "pre", dfs_root))
    dfs_post = _phase_rates(_derive_failure_path(dfs_summary, "post", dfs_root))
    phase_labels = ["Early", "Middle", "Late"]
    x = list(range(3))
    w = 0.35

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    axs[0].bar([i - w / 2 for i in x], bfs_pre, width=w, color=COLOR_BASELINE, label="Pre")
    axs[0].bar([i + w / 2 for i in x], bfs_post, width=w, color=COLOR_SFT, label="Post")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(phase_labels)
    axs[0].set_title("BFS")
    axs[0].set_ylabel("Failure rate")
    axs[0].legend(frameon=False)

    axs[1].bar([i - w / 2 for i in x], dfs_pre, width=w, color=COLOR_BASELINE, label="Pre")
    axs[1].bar([i + w / 2 for i in x], dfs_post, width=w, color=COLOR_SFT, label="Post")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(phase_labels)
    axs[1].set_title("DFS")

    for ax in axs:
        ax.set_ylim(0, 1.05)

    fig.suptitle("Failure phase profile shifts (pre vs post)")
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def fig_operation_failure_heatmap(
    bfs_summary: dict[str, Any],
    dfs_summary: dict[str, Any],
    bfs_root: Path,
    dfs_root: Path,
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    sources = {
        "BFS pre": _op_rates(_derive_failure_path(bfs_summary, "pre", bfs_root)),
        "BFS post": _op_rates(_derive_failure_path(bfs_summary, "post", bfs_root)),
        "DFS pre": _op_rates(_derive_failure_path(dfs_summary, "pre", dfs_root)),
        "DFS post": _op_rates(_derive_failure_path(dfs_summary, "post", dfs_root)),
    }
    ops = sorted({k for d in sources.values() for k in d.keys()})
    mat = np.array([[sources[row].get(op, np.nan) for op in ops] for row in sources.keys()], dtype=float)

    fig, ax = plt.subplots(figsize=(max(6.4, 0.95 * len(ops)), 3.8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(np.arange(len(ops)))
    ax.set_xticklabels(ops, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(sources)))
    ax.set_yticklabels(list(sources.keys()))
    ax.set_title("Failure rate by operation type")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Failure rate")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else "black")

    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def fig_benchmark_pre_post(
    bfs_summary: dict[str, Any],
    dfs_summary: dict[str, Any],
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = ["BFS", "DFS"]
    pre_nl = [_extract_metric(bfs_summary, "nlgraph_step_accuracy")[0], _extract_metric(dfs_summary, "nlgraph_step_accuracy")[0]]
    post_nl = [_extract_metric(bfs_summary, "nlgraph_step_accuracy")[1], _extract_metric(dfs_summary, "nlgraph_step_accuracy")[1]]
    pre_gl = [_extract_metric(bfs_summary, "glbench_step_accuracy")[0], _extract_metric(dfs_summary, "glbench_step_accuracy")[0]]
    post_gl = [_extract_metric(bfs_summary, "glbench_step_accuracy")[1], _extract_metric(dfs_summary, "glbench_step_accuracy")[1]]

    x = list(range(len(labels)))
    w = 0.34
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)

    axs[0].bar([i - w / 2 for i in x], pre_nl, width=w, color=COLOR_BASELINE, label="Pre")
    axs[0].bar([i + w / 2 for i in x], post_nl, width=w, color=COLOR_SFT, label="Post")
    axs[0].set_title("NLGraph step accuracy")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(frameon=False)

    axs[1].bar([i - w / 2 for i in x], pre_gl, width=w, color=COLOR_BASELINE, label="Pre")
    axs[1].bar([i + w / 2 for i in x], post_gl, width=w, color=COLOR_SFT, label="Post")
    axs[1].set_title("GLBench step accuracy")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)

    for ax in axs:
        ax.set_ylim(0, 1.05)

    fig.suptitle("Benchmark behavior before/after DAgger")
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def fig_top_confusion_mass(
    bfs_summary: dict[str, Any],
    dfs_summary: dict[str, Any],
    bfs_root: Path,
    dfs_root: Path,
    out_base: Path,
    top_k: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    series = {
        "BFS pre": _sum_top_confusions(_derive_failure_path(bfs_summary, "pre", bfs_root), top_k=top_k),
        "BFS post": _sum_top_confusions(_derive_failure_path(bfs_summary, "post", bfs_root), top_k=top_k),
        "DFS pre": _sum_top_confusions(_derive_failure_path(dfs_summary, "pre", dfs_root), top_k=top_k),
        "DFS post": _sum_top_confusions(_derive_failure_path(dfs_summary, "post", dfs_root), top_k=top_k),
    }
    labels = list(series.keys())
    vals = [series[k] for k in labels]
    colors = [COLOR_BASELINE, COLOR_SFT, COLOR_BASELINE, COLOR_SFT]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel(f"Sum of top-{top_k} confusion counts")
    ax.set_title("Concentration of repeated confusions")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfs-summary", required=True, help="Path to BFS summary_iterative_pre_post.json")
    ap.add_argument("--dfs-summary", required=True, help="Path to DFS summary_iterative_pre_post.json")
    ap.add_argument("--out-dir", default="paper/figures/cross_algo", help="Output directory for figure set")
    args = ap.parse_args()

    bfs_summary = _load(Path(args.bfs_summary))
    dfs_summary = _load(Path(args.dfs_summary))
    bfs_root = Path(args.bfs_summary).resolve().parent
    dfs_root = Path(args.dfs_summary).resolve().parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()

    fig_main_pre_post(bfs_summary, dfs_summary, out_dir / "fig_cross_main_pre_post")
    fig_delta_bars(bfs_summary, dfs_summary, out_dir / "fig_cross_delta_bars")
    fig_failure_phase_grid(bfs_summary, dfs_summary, bfs_root, dfs_root, out_dir / "fig_cross_failure_phase")
    fig_operation_failure_heatmap(
        bfs_summary,
        dfs_summary,
        bfs_root,
        dfs_root,
        out_dir / "fig_cross_op_failure_heatmap",
    )
    fig_benchmark_pre_post(bfs_summary, dfs_summary, out_dir / "fig_cross_benchmark_pre_post")
    fig_top_confusion_mass(
        bfs_summary,
        dfs_summary,
        bfs_root,
        dfs_root,
        out_dir / "fig_cross_confusion_mass",
    )

    print(f"Done. Wrote cross-algorithm figure set to {out_dir}")


if __name__ == "__main__":
    main()
