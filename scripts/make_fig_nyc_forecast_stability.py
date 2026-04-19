from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.style import OKABE_ITO, apply_journal_style, despine, nature_double_column, panel_label, save_figure, set_figure_facecolor


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot NYC time-machine forecast stability windows (days-to-forecast-stable).")
    ap.add_argument(
        "--stability",
        default="results/revision/nyc_time_machine_forecast_stability.tsv",
        help="Stability summary TSV (default: results/revision/nyc_time_machine_forecast_stability.tsv).",
    )
    ap.add_argument("--out-prefix", default="plots/fig_nyc_forecast_stability", help="Output prefix (no extension).")
    args = ap.parse_args()

    apply_journal_style()

    src = (REPO_ROOT / str(args.stability)).resolve()
    if not src.exists():
        raise SystemExit(f"missing stability table: {src}")

    df = pd.read_csv(src, sep="\t")
    req = {"metric", "horizon_weeks", "days_to_forecast_stable"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise SystemExit(f"stability table missing columns: {', '.join(miss)}")

    df["horizon_weeks"] = pd.to_numeric(df["horizon_weeks"], errors="coerce").astype(int)
    df["days_to_forecast_stable"] = pd.to_numeric(df["days_to_forecast_stable"], errors="coerce")
    df = df.dropna(subset=["metric", "days_to_forecast_stable"]).copy()
    if df.empty:
        raise SystemExit("no stability rows to plot")

    metrics = sorted(df["metric"].astype(str).unique().tolist())
    horizons = sorted(df["horizon_weeks"].unique().tolist())

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    size = nature_double_column(height_in=3.2)
    fig = plt.figure(figsize=(size.width_in, size.height_in), constrained_layout=True)
    set_figure_facecolor(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    # Violin/box can be unstable with few points; use scatter + median bars.
    colors = [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["vermillion"], OKABE_ITO["purple"]]
    for i, m in enumerate(metrics):
        sub = df[df["metric"].astype(str) == m].copy()
        for j, h in enumerate(horizons):
            s = sub[sub["horizon_weeks"] == h]["days_to_forecast_stable"].dropna()
            if s.empty:
                continue
            x = j + (i - (len(metrics) - 1) / 2) * 0.12
            ax0.scatter([x] * len(s), s, color=colors[i % len(colors)], alpha=0.85, s=14, label=m if j == 0 else None, linewidths=0)
            ax0.plot([x - 0.05, x + 0.05], [s.median(), s.median()], color=colors[i % len(colors)], linewidth=2.0)

    ax0.set_xticks(range(len(horizons)))
    ax0.set_xticklabels([f"Lead {h}w" for h in horizons])
    ax0.set_ylabel("Days-to-forecast-stable")
    ax0.set_title("Forecast stability")
    ax0.grid(True, axis="y")
    despine(ax0)
    ax0.legend(loc="upper right")
    panel_label(ax0, "A")

    ax1 = fig.add_subplot(gs[0, 1])
    # Show the fraction stabilized by day thresholds for a simple “certainty window”.
    thresholds = [0, 7, 14, 21, 28]
    lines = []
    for i, m in enumerate(metrics):
        sub = df[df["metric"].astype(str) == m].copy()
        y = []
        for t in thresholds:
            y.append(float((sub["days_to_forecast_stable"] <= t).mean()) if len(sub) else float("nan"))
        ax1.plot(thresholds, y, marker="o", color=colors[i % len(colors)], label=m)
        lines.append((m, y))
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Days after first commit in window")
    ax1.set_ylabel("Fraction stabilized")
    ax1.set_title("Stabilized-by-day curve")
    ax1.grid(True)
    despine(ax1)
    ax1.legend(loc="lower right")
    panel_label(ax1, "B")

    out_png, out_pdf = save_figure(fig, out_prefix=str(out_prefix), dpi=600)
    plt.close(fig)
    print(f"Wrote figure: {out_png}")
    print(f"Wrote figure: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
