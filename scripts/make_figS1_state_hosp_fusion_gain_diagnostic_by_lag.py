from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.style import OKABE_ITO
from figures.style import apply_journal_style
from figures.style import despine
from figures.style import nature_double_column
from figures.style import panel_label
from figures.style import save_figure
from figures.style import set_figure_facecolor


REPO_ROOT = Path(__file__).resolve().parents[1]


PATHOGEN_STYLE = {
    "COVID-19": {"color": OKABE_ITO["blue"], "marker": "o", "label": "COVID-19"},
    "Influenza": {"color": OKABE_ITO["orange"], "marker": "s", "label": "Influenza"},
    "RSV": {"color": OKABE_ITO["green"], "marker": "^", "label": "RSV"},
}


def _scatter(ax: plt.Axes, df: pd.DataFrame, *, title: str) -> None:
    ax.axhline(0, color=OKABE_ITO["gray"], linewidth=0.8, zorder=0)
    ax.axvline(0, color=OKABE_ITO["gray"], linewidth=0.8, zorder=0)

    for pathogen, g in df.groupby("pathogen"):
        st = PATHOGEN_STYLE.get(str(pathogen), {"color": OKABE_ITO["black"], "marker": "o", "label": str(pathogen)})
        ax.scatter(
            g["best_lag_weeks"],
            g["delta_rmse_fusion_minus_ar"],
            s=14,
            alpha=0.8,
            linewidths=0.0,
            c=st["color"],
            marker=st["marker"],
            label=st["label"],
        )

    ax.set_title(title)
    ax.set_xlabel("Best lag (weeks): wastewater leads if >0")
    ax.set_ylabel("ΔRMSE (fusion − admissions-only)")
    ax.grid(True, axis="both")
    despine(ax)


def main() -> int:
    ap = argparse.ArgumentParser(description="Figure S1: lead–lag association versus per-series fusion gain (state scale; horizon=1).")
    ap.add_argument(
        "--table",
        default="results/diagnostics/state_hosp_fusion_gain_diagnostic_by_lag.per100k.tsv",
        help="Diagnostic table TSV (default: results/diagnostics/state_hosp_fusion_gain_diagnostic_by_lag.per100k.tsv).",
    )
    ap.add_argument(
        "--out-prefix",
        default="plots/FigureS1_state_hosp_fusion_gain_by_lag",
        help="Output prefix for png/pdf (default: plots/FigureS1_state_hosp_fusion_gain_by_lag).",
    )
    args = ap.parse_args()

    table_path = (REPO_ROOT / str(args.table)).resolve()
    if not table_path.exists():
        raise SystemExit(f"missing table: {table_path}")

    df = pd.read_csv(table_path, sep="\t")
    need = {"context", "horizon_weeks", "pathogen", "best_lag_weeks", "delta_rmse_fusion_minus_ar"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise SystemExit(f"table missing columns: {', '.join(miss)}")

    df = df.copy()
    df["horizon_weeks"] = pd.to_numeric(df["horizon_weeks"], errors="coerce")
    df["best_lag_weeks"] = pd.to_numeric(df["best_lag_weeks"], errors="coerce")
    df["delta_rmse_fusion_minus_ar"] = pd.to_numeric(df["delta_rmse_fusion_minus_ar"], errors="coerce")
    df = df[df["horizon_weeks"] == 1].copy()
    df = df.dropna(subset=["best_lag_weeks", "delta_rmse_fusion_minus_ar"]).copy()

    apply_journal_style()
    size = nature_double_column(height_in=2.6)
    fig, axes = plt.subplots(1, 2, figsize=(size.width_in, size.height_in), sharey=True)
    set_figure_facecolor(fig)

    left = df[df["context"] == "early_warning"].copy()
    right = df[df["context"] == "with_current_y"].copy()

    _scatter(axes[0], left, title="Early warning (no contemporaneous admissions)")
    _scatter(axes[1], right, title="Nowcasting (with contemporaneous admissions)")

    panel_label(axes[0], "A")
    panel_label(axes[1], "B")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png, out_pdf = save_figure(fig, out_prefix=str(out_prefix), dpi=600)
    plt.close(fig)
    print(f"OK figS1: wrote {Path(out_png).relative_to(REPO_ROOT)} and {Path(out_pdf).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
