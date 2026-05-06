"""
Build paper figures from committed JSON under paper/r1_n10_metrics/ (Qwen n=10, c100, s100).

Run from repo root:
  python -m plots.paper_r1_n10_figures --out-dir paper/figures/r1_n10_qwen
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_json(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def _step_acc(d: dict) -> float:
    return float(d.get("step_accuracy", d.get("overall_step_accuracy", 0.0)))


def _save_both(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suf in (".pdf", ".png"):
        p = base.with_suffix(suf)
        fig.savefig(p, bbox_inches="tight")
        print(f"Wrote {p}")


def fig_step_accuracy_bundle(*, root: Path, out_base: Path) -> None:
    import matplotlib.pyplot as plt

    from plots.style import apply_paper_style

    apply_paper_style()
    bfs = root / "bfs_pre_sft"
    dfs = root / "dfs_pre_sft"
    tag_b = "bfs_erdos_renyi_n10_c100_s100"
    tag_d = "dfs_erdos_renyi_n10_c100_s100"

    main_b = _step_acc(_load_json(bfs / f"metrics_{tag_b}.json"))
    main_d = _step_acc(_load_json(dfs / f"metrics_{tag_d}.json"))
    nl_b = _step_acc(_load_json(bfs / f"nlgraph_{tag_b}_operation_accuracy.json"))
    nl_d = _step_acc(_load_json(dfs / f"nlgraph_{tag_d}_operation_accuracy.json"))
    gl_path_b = bfs / f"glbench_{tag_b}_operation_accuracy.json"
    gl_path_d = dfs / f"glbench_{tag_d}_operation_accuracy.json"
    gl_b = _step_acc(_load_json(gl_path_b)) if gl_path_b.is_file() else float("nan")
    gl_d = _step_acc(_load_json(gl_path_d)) if gl_path_d.is_file() else float("nan")

    labels = [
        "Pipeline\nBFS",
        "Pipeline\nDFS",
        "NLGraph\nBFS",
        "NLGraph\nDFS",
        "GLBench\nBFS",
        "GLBench\nDFS",
    ]
    vals = [main_b, main_d, nl_b, nl_d, gl_b, gl_d]
    colors = ["#0072B2", "#D55E00", "#56B4E9", "#E69F00", "#6A994E", "#009E73"]

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = range(len(labels))
    bars = ax.bar(list(x), vals, color=colors, edgecolor="#333333", linewidth=0.5)
    for i, (v, b) in enumerate(zip(vals, bars)):
        if v != v:  # NaN
            b.set_height(0)
            ax.text(i, 0.02, "n/a", ha="center", fontsize=8, color="#666666")
        else:
            ax.text(i, min(1.02, v + 0.03), f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Step accuracy")
    ax.set_title("Qwen2.5-7B SFT (pre–DAgger round 1): n=10, 100 pipeline graphs")
    fig.tight_layout()
    _save_both(fig, out_base / "fig_r1_step_accuracy_main_and_benchmarks")
    plt.close(fig)


def _run_plot_script(module: str, argv: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *argv]
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("paper/r1_n10_metrics"),
        help="Root with bfs_pre_sft/ and dfs_pre_sft/",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("paper/figures/r1_n10_qwen"))
    args = ap.parse_args()

    root = args.metrics_root.resolve()
    out = args.out_dir.resolve()
    if not (root / "bfs_pre_sft").is_dir():
        raise FileNotFoundError(f"Missing {root / 'bfs_pre_sft'}")

    fig_step_accuracy_bundle(root=root, out_base=out)

    tag_b = "bfs_erdos_renyi_n10_c100_s100"
    tag_d = "dfs_erdos_renyi_n10_c100_s100"
    bf = root / "bfs_pre_sft" / f"failures_{tag_b}.json"
    df = root / "dfs_pre_sft" / f"failures_{tag_d}.json"

    png_op = out / "fig_r1_failure_by_operation_type.png"
    png_fe = out / "fig_r1_first_error_distribution.png"
    _run_plot_script(
        "plots.failure_by_operation_type",
        [
            "--inputs",
            str(bf),
            str(df),
            "--labels",
            "BFS (SFT)",
            "DFS (SFT)",
            "--out",
            str(png_op),
        ],
    )
    _run_plot_script(
        "plots.first_error_distribution",
        [
            "--inputs",
            str(bf),
            str(df),
            "--labels",
            "BFS (SFT)",
            "DFS (SFT)",
            "--out",
            str(png_fe),
        ],
    )

    # Duplicate PNGs as PDFs for LaTeX (vector-ish via matplotlib re-save optional; use png in tex or convert)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        for png_path in (png_op, png_fe):
            img = mpimg.imread(png_path)
            fig, ax = plt.subplots(figsize=(img.shape[1] / 200, img.shape[0] / 200), dpi=200)
            ax.imshow(img)
            ax.axis("off")
            pdf_path = png_path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"Wrote {pdf_path}")
    except Exception as e:
        print(f"[warn] Could not rasterize PNG→PDF: {e}; use PNG in LaTeX or includepdf")


if __name__ == "__main__":
    main()
