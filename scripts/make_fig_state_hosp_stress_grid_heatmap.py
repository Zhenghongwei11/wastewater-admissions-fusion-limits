from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.style import OKABE_ITO, apply_journal_style, despine, nature_double_column, panel_label, save_figure, set_figure_facecolor


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot a compact heatmap from Route A stress grid overall table (early-warning focus).")
    ap.add_argument(
        "--overall",
        default="results/stress/state_hosp_stress_grid_overall.per100k.tsv",
        help="Overall stress grid table (default: results/stress/state_hosp_stress_grid_overall.per100k.tsv).",
    )
    ap.add_argument("--context", default="early_warning", help="Context to plot (default: early_warning).")
    ap.add_argument("--revision-frac", type=float, default=0.30, help="Revision fraction slice to plot (default: 0.30).")
    ap.add_argument(
        "--out-prefix",
        default="plots/figS_state_hosp_stress_grid_heatmap",
        help="Output prefix (no extension).",
    )
    args = ap.parse_args()

    apply_journal_style()

    src = (REPO_ROOT / str(args.overall)).resolve()
    if not src.exists():
        raise SystemExit(f"missing overall stress grid table: {src}")

    df = pd.read_csv(src, sep="\t")
    req = {
        "y_delay_weeks",
        "feature_missing_frac",
        "feature_revision_frac",
        "context",
        "n_strata",
        "total_series",
        "mean_delta_rmse_fusion_minus_ar",
        "mean_frac_fusion_better_than_ar",
    }
    miss = sorted(req - set(df.columns))
    if miss:
        raise SystemExit(f"overall table missing columns: {', '.join(miss)}")

    df["y_delay_weeks"] = pd.to_numeric(df["y_delay_weeks"], errors="coerce").astype(int)
    df["feature_missing_frac"] = pd.to_numeric(df["feature_missing_frac"], errors="coerce")
    df["feature_revision_frac"] = pd.to_numeric(df["feature_revision_frac"], errors="coerce")
    df["n_strata"] = pd.to_numeric(df["n_strata"], errors="coerce").fillna(0).astype(int)
    df["total_series"] = pd.to_numeric(df["total_series"], errors="coerce").fillna(0).astype(int)
    df["mean_delta_rmse_fusion_minus_ar"] = pd.to_numeric(df["mean_delta_rmse_fusion_minus_ar"], errors="coerce")
    df["mean_frac_fusion_better_than_ar"] = pd.to_numeric(df["mean_frac_fusion_better_than_ar"], errors="coerce")

    sub = df[(df["context"].astype(str) == str(args.context)) & (df["feature_revision_frac"] == float(args.revision_frac))].copy()
    if sub.empty:
        raise SystemExit("no rows for requested context/revision-frac slice")

    delays = sorted(sub["y_delay_weeks"].unique().tolist())
    missings = sorted(sub["feature_missing_frac"].unique().tolist())

    # Build matrices: delta + win-rate
    mat_delta = []
    mat_win = []
    for m in missings:
        row_d = []
        row_w = []
        for d in delays:
            r = sub[(sub["feature_missing_frac"] == m) & (sub["y_delay_weeks"] == d)]
            if r.empty:
                row_d.append(math.nan)
                row_w.append(math.nan)
            else:
                row_d.append(float(r["mean_delta_rmse_fusion_minus_ar"].iloc[0]))
                row_w.append(float(r["mean_frac_fusion_better_than_ar"].iloc[0]))
        mat_delta.append(row_d)
        mat_win.append(row_w)

    finite = [abs(v) for row in mat_delta for v in row if isinstance(v, float) and math.isfinite(v)]
    vmax = max(finite) if finite else 1e-6
    vmax = max(vmax, 1e-6)

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    size = nature_double_column(height_in=3.0)
    fig = plt.figure(figsize=(size.width_in, size.height_in), constrained_layout=True)
    set_figure_facecolor(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mat_delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    title_ctx = "Early warning" if str(args.context) == "early_warning" else ("Nowcasting" if str(args.context) == "with_current_y" else str(args.context))
    ax.set_title(f"Sensitivity to reporting constraints (revision probability = {args.revision_frac:g}; {title_ctx})")
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels([str(d) for d in delays])
    ax.set_xlabel("Reporting delay (weeks)")
    ax.set_yticks(range(len(missings)))
    ax.set_yticklabels([f"{m:.2f}" for m in missings])
    ax.set_ylabel("Covariate missingness (fraction)")
    despine(ax)

    for i in range(len(missings)):
        for j in range(len(delays)):
            v = mat_delta[i][j]
            if not (isinstance(v, float) and math.isfinite(v)):
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.6, color=OKABE_ITO["black"])

    panel_label(ax, "A")

    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mean ΔRMSE (fusion − admissions-only)")

    out_png, out_pdf = save_figure(fig, out_prefix=str(out_prefix), dpi=600)
    plt.close(fig)
    print(f"Wrote figure: {out_png}")
    print(f"Wrote figure: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
