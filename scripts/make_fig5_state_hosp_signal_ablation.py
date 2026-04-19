from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.style import OKABE_ITO, apply_journal_style, despine, nature_double_column, panel_label, save_figure, set_figure_facecolor


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Make Figure 5 (Route A): state hospitalization signal ablation summary (fusion vs baselines).")
    ap.add_argument(
        "--summary",
        default="results/ablation/state_hosp_ablation_summary.tsv",
        help="Ablation summary TSV input.",
    )
    ap.add_argument("--out-prefix", default="plots/fig5_state_hosp_signal_ablation", help="Output prefix (no extension).")
    args = ap.parse_args()

    apply_journal_style()

    path = (REPO_ROOT / str(args.summary)).resolve()
    if not path.exists():
        raise SystemExit(f"missing ablation summary: {path}")

    df = pd.read_csv(path, sep="\t")
    req = {"context", "pathogen", "horizon_weeks", "comparison", "median_delta_rmse", "frac_negative_delta_rmse"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise SystemExit(f"ablation summary missing columns: {', '.join(miss)}")

    df = df[df["comparison"].astype(str) == "ridge_fusion_minus_ridge_ar"].copy()
    df["horizon_weeks"] = pd.to_numeric(df["horizon_weeks"], errors="coerce").astype(int)
    df["median_delta_rmse"] = pd.to_numeric(df["median_delta_rmse"], errors="coerce")
    df["frac_negative_delta_rmse"] = pd.to_numeric(df["frac_negative_delta_rmse"], errors="coerce")

    contexts = [c for c in ["early_warning", "with_current_y"] if (df["context"].astype(str) == c).any()]
    pathogens = [p for p in ["COVID-19", "Influenza", "RSV"] if (df["pathogen"].astype(str) == p).any()]
    horizons = sorted(df["horizon_weeks"].unique().tolist())
    if not horizons:
        raise SystemExit("no horizons to plot")

    mats = {}
    fracs = {}
    for ctx in contexts:
        mat = []
        frac = []
        for p in pathogens:
            row = []
            rowf = []
            for h in horizons:
                sub = df[(df["context"] == ctx) & (df["pathogen"] == p) & (df["horizon_weeks"] == h)]
                if sub.empty:
                    row.append(math.nan)
                    rowf.append(math.nan)
                else:
                    row.append(float(sub["median_delta_rmse"].iloc[0]))
                    rowf.append(float(sub["frac_negative_delta_rmse"].iloc[0]))
            mat.append(row)
            frac.append(rowf)
        mats[ctx] = mat
        fracs[ctx] = frac

    finite = [abs(v) for ctx in contexts for row in mats[ctx] for v in row if isinstance(v, float) and math.isfinite(v)]
    vmax = max(finite) if finite else 1e-6
    vmax = max(vmax, 1e-6)

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    size = nature_double_column(height_in=3.2)
    fig = plt.figure(figsize=(size.width_in, size.height_in), constrained_layout=True)
    set_figure_facecolor(fig)
    gs = fig.add_gridspec(1, len(contexts) + 1, width_ratios=[1.0] * len(contexts) + [0.05], wspace=0.25)

    for i, ctx in enumerate(contexts):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(mats[ctx], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        title = "Early warning" if ctx == "early_warning" else ("Nowcasting" if ctx == "with_current_y" else ctx)
        ax.set_title(title)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("Lead time (weeks)")
        ax.set_yticks(range(len(pathogens)))
        ax.set_yticklabels(pathogens)
        ax.set_ylabel("Pathogen")
        despine(ax)

        for r in range(len(pathogens)):
            for c in range(len(horizons)):
                v = mats[ctx][r][c]
                w = fracs[ctx][r][c]
                if not isinstance(v, float) or not math.isfinite(v):
                    continue
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7.8, color=OKABE_ITO["black"])

        panel_label(ax, chr(ord("A") + i))

    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Median ΔRMSE (fusion − baseline)\n(negative values favor fusion)")

    out_png, out_pdf = save_figure(fig, out_prefix=str(out_prefix), dpi=600)
    plt.close(fig)
    print(f"Wrote figure: {out_png}")
    print(f"Wrote figure: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
