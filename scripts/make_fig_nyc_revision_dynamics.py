from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.style import OKABE_ITO, apply_journal_style, despine, nature_double_column, panel_label, save_figure, set_figure_facecolor


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot NYC git-based revision dynamics (days-to-stable, distinct value counts).")
    ap.add_argument(
        "--summary",
        default="results/revision/nyc_ed_respiratory_illness_git_revision_summary.tsv",
        help="Revision summary TSV (default: results/revision/nyc_ed_respiratory_illness_git_revision_summary.tsv).",
    )
    ap.add_argument("--out-prefix", default="plots/fig_nyc_revision_dynamics", help="Output prefix (no extension).")
    args = ap.parse_args()

    apply_journal_style()

    src = (REPO_ROOT / str(args.summary)).resolve()
    if not src.exists():
        raise SystemExit(f"missing revision summary: {src}")

    df = pd.read_csv(src, sep="\t")
    req = {"week_end", "metric", "n_distinct_values", "days_to_stable"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise SystemExit(f"revision summary missing columns: {', '.join(miss)}")

    df["n_distinct_values"] = pd.to_numeric(df["n_distinct_values"], errors="coerce")
    df["days_to_stable"] = pd.to_numeric(df["days_to_stable"], errors="coerce")
    df = df.dropna(subset=["metric"]).copy()

    metrics = sorted(df["metric"].astype(str).unique().tolist())
    if not metrics:
        raise SystemExit("no metrics to plot")

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    size = nature_double_column(height_in=3.1)
    fig = plt.figure(figsize=(size.width_in, size.height_in), constrained_layout=True)
    set_figure_facecolor(fig)
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    for m in metrics:
        s = df[df["metric"].astype(str) == m]["days_to_stable"].dropna()
        if s.empty:
            continue
        ax0.hist(s, bins=15, alpha=0.55, label=m, edgecolor="white", linewidth=0.6)
    ax0.set_title("Published-value stability (days)")
    ax0.set_xlabel("Days until last change")
    ax0.set_ylabel("Number of weeks")
    ax0.grid(True, axis="y")
    despine(ax0)
    ax0.legend(loc="upper right")
    panel_label(ax0, "A")

    ax1 = fig.add_subplot(gs[0, 1])
    box_data = []
    box_labels = []
    for m in metrics:
        s = df[df["metric"].astype(str) == m]["n_distinct_values"].dropna()
        if s.empty:
            continue
        box_data.append(s.to_numpy())
        box_labels.append(m)
    ax1.boxplot(
        box_data,
        tick_labels=box_labels,
        vert=True,
        showfliers=False,
        boxprops={"color": OKABE_ITO["black"], "linewidth": 0.9},
        medianprops={"color": OKABE_ITO["vermillion"], "linewidth": 1.4},
        whiskerprops={"color": OKABE_ITO["black"], "linewidth": 0.9},
        capprops={"color": OKABE_ITO["black"], "linewidth": 0.9},
    )
    ax1.set_title("Revision multiplicity (distinct values)")
    ax1.set_ylabel("# distinct values observed across commits")
    ax1.grid(True, axis="y")
    ax1.tick_params(axis="x", rotation=20)
    despine(ax1)
    panel_label(ax1, "B")

    out_png, out_pdf = save_figure(fig, out_prefix=str(out_prefix), dpi=600)
    plt.close(fig)
    print(f"Wrote figure: {out_png}")
    print(f"Wrote figure: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
