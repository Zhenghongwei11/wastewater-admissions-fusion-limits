from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.style import OKABE_ITO, apply_journal_style, despine, nature_double_column, panel_label, save_figure, set_figure_facecolor


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Make Figure 2 (Route A): state-level wastewater ↔ hospital admissions lead/lag.")
    ap.add_argument(
        "--leadlag",
        default="results/leadlag/wastewater_hosp_leadlag.geo_matched_state.tsv",
        help="Lead/lag TSV input.",
    )
    ap.add_argument("--out-prefix", default="plots/fig2_state_ww_hosp_leadlag", help="Output prefix (no extension).")
    args = ap.parse_args()

    apply_journal_style()

    path = (REPO_ROOT / str(args.leadlag)).resolve()
    if not path.exists():
        raise SystemExit(f"missing leadlag table: {path}")

    df = pd.read_csv(path, sep="\t")
    req = {"geo_id", "pathogen", "best_lag_weeks"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise SystemExit(f"leadlag missing columns: {', '.join(miss)}")

    df["best_lag_weeks"] = pd.to_numeric(df["best_lag_weeks"], errors="coerce")
    df = df.dropna(subset=["best_lag_weeks"]).copy()
    df["best_lag_weeks"] = df["best_lag_weeks"].astype(int)

    pathogens = [p for p in ["COVID-19", "Influenza", "RSV"] if (df["pathogen"].astype(str) == p).any()]
    if not pathogens:
        raise SystemExit("no pathogens to plot")

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    size = nature_double_column(height_in=2.9)
    fig = plt.figure(figsize=(size.width_in, size.height_in), constrained_layout=True)
    set_figure_facecolor(fig)
    gs = fig.add_gridspec(1, len(pathogens), wspace=0.25)

    for j, pathogen in enumerate(pathogens):
        ax = fig.add_subplot(gs[0, j])
        sub = df[df["pathogen"] == pathogen].copy()
        if sub.empty:
            continue
        lags = sub["best_lag_weeks"].astype(int)
        bins = list(range(int(lags.min()) - 1, int(lags.max()) + 2))
        ax.hist(lags, bins=bins, color=OKABE_ITO["sky"], alpha=0.95, edgecolor="#FFFFFF", linewidth=0.6)
        ax.set_title(f"{pathogen}", fontsize=9)
        ax.set_xlabel("Best lag (weeks)")
        ax.set_ylabel("Number of geographies")
        ax.grid(True, axis="y")
        despine(ax)

        med = float(lags.median())
        q1 = float(lags.quantile(0.25))
        q3 = float(lags.quantile(0.75))
        ax.axvline(med, color=OKABE_ITO["vermillion"], linewidth=1.8, label="Median")
        ax.text(
            0.98,
            0.98,
            f"N={len(sub)}\nMedian={med:+.0f}\nIQR=[{q1:+.0f},{q3:+.0f}]",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.95, "edgecolor": "#D0D0D0"},
        )
        if j == 0:
            ax.legend(loc="upper left")
        panel_label(ax, chr(ord("A") + j))

    out_png, out_pdf = save_figure(fig, out_prefix=str(out_prefix), dpi=600)
    plt.close(fig)
    print(f"Wrote figure: {out_png}")
    print(f"Wrote figure: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
