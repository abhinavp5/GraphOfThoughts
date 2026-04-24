"""
Generate all paper-ready figures from pipeline metric JSONs.

Run from the repository root:

  python -m plots.generate_paper_figures \\
      --tag bfs_erdos_renyi_n10_c5_s100 \\
      --baseline-dir out \\
      --sft-dir out_sft \\
      --out-dir paper/figures

Writes matching .png and .pdf for each figure (vector PDF for LaTeX).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plots.style import COLOR_BASELINE, COLOR_SFT, apply_paper_style


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _step_acc(d: dict) -> float:
    return float(d.get("step_accuracy", d.get("overall_step_accuracy", 0.0)))


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        p = out_base.with_suffix(suffix)
        fig.savefig(p, bbox_inches="tight")
        print(f"Wrote {p}")


def fig_step_accuracy_overview(
    *,
    baseline_dir: Path,
    sft_dir: Path,
    tag: str,
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt

    main_b = baseline_dir / f"metrics_{tag}.json"
    main_s = sft_dir / f"metrics_{tag}.json"
    nl_b = baseline_dir / f"nlgraph_{tag}_operation_accuracy.json"
    nl_s = sft_dir / f"nlgraph_{tag}_operation_accuracy.json"
    gl_b = baseline_dir / f"glbench_{tag}_operation_accuracy.json"
    gl_s = sft_dir / f"glbench_{tag}_operation_accuracy.json"

    groups: list[str] = []
    b_vals: list[float] = []
    s_vals: list[float] = []

    def add_group(label: str, pb: Path, ps: Path, allow_missing: bool = False) -> None:
        if not pb.exists() or not ps.exists():
            if allow_missing:
                return
            raise FileNotFoundError(f"Missing metrics for group {label}: {pb} / {ps}")
        db = _load(pb)
        ds = _load(ps)
        if db.get("n_samples") == 0 and allow_missing:
            return
        if ds.get("n_samples") == 0 and allow_missing:
            return
        groups.append(label)
        b_vals.append(_step_acc(db))
        s_vals.append(_step_acc(ds))

    add_group("Main trace", main_b, main_s)
    add_group("NLGraph", nl_b, nl_s, allow_missing=True)
    add_group("GLBench", gl_b, gl_s, allow_missing=True)

    if not groups:
        raise RuntimeError("No metric groups found to plot.")

    x = range(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([i - w / 2 for i in x], b_vals, width=w, label="Baseline", color=COLOR_BASELINE)
    ax.bar([i + w / 2 for i in x], s_vals, width=w, label="SFT", color=COLOR_SFT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Step accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Step accuracy: baseline vs SFT")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for i in x:
        ax.text(i - w / 2, b_vals[i] + 0.02, f"{b_vals[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, s_vals[i] + 0.02, f"{s_vals[i]:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def _first_errors(path: Path) -> list[int]:
    d = _load(path)
    out: list[int] = []
    for r in d.get("per_sample", []):
        x = r.get("first_error_step")
        if x is not None:
            out.append(int(x))
    return out


def fig_first_error(
    *,
    baseline_failures: Path,
    sft_failures: Path,
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    all_vals: list[int] = []
    for path, label, color in (
        (baseline_failures, "Baseline", COLOR_BASELINE),
        (sft_failures, "SFT", COLOR_SFT),
    ):
        vals = _first_errors(path)
        all_vals.extend(vals)
        if vals:
            hi = max(vals) + 1
            bins = np.arange(-0.5, hi + 0.5, 1.0)
            ax.hist(vals, bins=bins, alpha=0.55, label=label, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("First incorrect step index")
    ax.set_ylabel("Number of traces")
    ax.set_title("Distribution of first error depth")
    if all_vals:
        ax.set_xlim(-0.5, max(all_vals) + 0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def _failure_rates_by_op(path: Path) -> dict[str, float]:
    d = _load(path)
    return {k: float(v) for k, v in d.get("failure_rate_by_gold_operation_type", {}).items()}


def fig_failure_by_operation(
    *,
    baseline_failures: Path,
    sft_failures: Path,
    out_base: Path,
    top_k: int = 8,
) -> None:
    import matplotlib.pyplot as plt

    runs = [
        ("Baseline", _failure_rates_by_op(baseline_failures)),
        ("SFT", _failure_rates_by_op(sft_failures)),
    ]
    ops = sorted({op for _, r in runs for op in r.keys()})
    avg = {op: sum(r.get(op, 0.0) for _, r in runs) / len(runs) for op in ops}
    ops = sorted(ops, key=lambda x: avg[x], reverse=True)[:top_k]
    if not ops:
        return

    width = 0.8 / len(runs)
    xs = list(range(len(ops)))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [COLOR_BASELINE, COLOR_SFT]
    for i, ((label, r), c) in enumerate(zip(runs, colors)):
        offsets = [x - 0.4 + width / 2 + i * width for x in xs]
        vals = [r.get(op, 0.0) for op in ops]
        ax.bar(offsets, vals, width=width, label=label, color=c)

    ax.set_xticks(xs)
    ax.set_xticklabels(ops, rotation=28, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Failure rate (by gold op type)")
    ax.set_title("Where the model fails: operation type")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def fig_failure_phase(
    *,
    baseline_failures: Path,
    sft_failures: Path,
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt

    order = ("early", "middle", "late")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    w = 0.36
    x = range(len(order))

    def series(path: Path) -> list[float]:
        d = _load(path)
        fr = d.get("failure_rate_by_step_bucket", {})
        return [float(fr.get(k, 0.0)) for k in order]

    b = series(baseline_failures)
    s = series(sft_failures)
    ax.bar([i - w / 2 for i in x], b, width=w, label="Baseline", color=COLOR_BASELINE)
    ax.bar([i + w / 2 for i in x], s, width=w, label="SFT", color=COLOR_SFT)
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.capitalize() for k in order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Step failure rate")
    ax.set_title("Failures by trace phase (early / middle / late)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def fig_top_confusions(
    failures_path: Path,
    out_base: Path,
    title_suffix: str,
    k: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    d = _load(failures_path)
    rows = d.get("top_operation_confusions", [])[:k]
    if not rows:
        return
    labels = [f"{r['gold_operation'][:28]} → {str(r['predicted_operation'])[:24]}" for r in rows]
    counts = [int(r["count"]) for r in rows]
    y = range(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.35 * len(labels) + 1.2)))
    ax.barh(list(y), counts, color=COLOR_SFT)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top gold→predicted confusions ({title_suffix})")
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def fig_structural(
    structural_json: Path,
    out_base: Path,
) -> None:
    import matplotlib.pyplot as plt

    d = _load(structural_json)
    by_n = d.get("by_graph_size", {})
    train_max_n = int(d.get("train_max_n", 0))

    xs = sorted(int(k) for k in by_n.keys())
    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    if len(xs) >= 2:
        ys = [float(by_n[str(n)]["step_accuracy"]) for n in xs]
        ax.plot(xs, ys, marker="o", color=COLOR_SFT, linewidth=2)
        ax.set_xticks(xs)
        ax.set_xlabel("Graph size n")
        ax.set_ylabel("Step accuracy")
        ax.set_title("Structural generalization (by n)")
        ax.set_ylim(0, 1.05)
        if train_max_n > 0:
            ax.axvline(train_max_n, linestyle="--", color="#666666", linewidth=1.2, label=f"Train max n = {train_max_n}")
            ax.legend(frameon=False, loc="best")
    else:
        id_acc = float(d.get("in_distribution_step_accuracy", 0.0))
        ood_acc = float(d.get("out_of_distribution_step_accuracy", 0.0))
        ood_steps = int(d.get("grouped", {}).get("out_of_distribution", {}).get("steps", 0))
        if ood_steps > 0:
            labels = ["In-distribution\n(n ≤ train max)", "Out-of-distribution\n(n > train max)"]
            vals = [id_acc, ood_acc]
            colors = [COLOR_BASELINE, COLOR_SFT]
        else:
            labels = ["In-distribution\n(eval graphs at train n)"]
            vals = [id_acc]
            colors = [COLOR_SFT]
        xb = range(len(labels))
        ax.bar(list(xb), vals, color=colors, edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(list(xb))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Step accuracy")
        ax.set_title("Structural generalization (aggregate)")
        note = f"Train max n = {train_max_n}."
        if len(xs) < 2:
            note += " Add eval at multiple n for a per-size curve."
        if ood_steps == 0:
            note += " No OOD-sized graphs in this eval set."
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom", color="#444444")

    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="Run tag, e.g. bfs_erdos_renyi_n10_c5_s100")
    ap.add_argument("--baseline-dir", type=Path, default=Path("out"), help="Directory with baseline JSONs")
    ap.add_argument("--sft-dir", type=Path, default=Path("out_sft"), help="Directory with SFT JSONs")
    ap.add_argument("--out-dir", type=Path, default=Path("paper/figures"), help="Figure output directory")
    args = ap.parse_args()

    apply_paper_style()
    import matplotlib.pyplot as plt

    root = Path.cwd()
    bdir = (root / args.baseline_dir).resolve()
    sdir = (root / args.sft_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    tag = args.tag

    bf = bdir / f"failures_{tag}.json"
    sf = sdir / f"failures_{tag}.json"
    if not bf.is_file() or not sf.is_file():
        raise FileNotFoundError(f"Need failures JSON for tag {tag} in both dirs:\n  {bf}\n  {sf}")

    fig_step_accuracy_overview(
        baseline_dir=bdir,
        sft_dir=sdir,
        tag=tag,
        out_base=out_dir / "fig_step_accuracy_overview",
    )
    fig_first_error(
        baseline_failures=bf,
        sft_failures=sf,
        out_base=out_dir / "fig_first_error_distribution",
    )
    fig_failure_by_operation(
        baseline_failures=bf,
        sft_failures=sf,
        out_base=out_dir / "fig_failure_by_operation_type",
    )
    fig_failure_phase(
        baseline_failures=bf,
        sft_failures=sf,
        out_base=out_dir / "fig_failure_by_phase",
    )
    fig_top_confusions(bf, out_dir / "fig_top_confusions_baseline", "baseline")
    fig_top_confusions(sf, out_dir / "fig_top_confusions_sft", "SFT")

    struct_s = sdir / f"structural_generalization_{tag}.json"
    if struct_s.is_file():
        fig_structural(struct_s, out_dir / "fig_structural_generalization")
    else:
        print(f"[skip] No structural generalization JSON: {struct_s}")

    plt.close("all")
    print(f"Done. Figures in {out_dir}")


if __name__ == "__main__":
    main()
