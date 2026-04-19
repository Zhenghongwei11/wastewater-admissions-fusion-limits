from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.style import OKABE_ITO, apply_journal_style, despine, nature_double_column, panel_label, save_figure, set_figure_facecolor


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Make Figure 4 (Route A): state hospitalization forecast benchmark summary.")
    ap.add_argument(
        "--summary",
        default="results/benchmark/state_hosp_forecast_benchmark_summary.tsv",
        help="Benchmark summary TSV input.",
    )
    ap.add_argument("--out-prefix", default="plots/fig4_state_hosp_forecast_benchmark", help="Output prefix (no extension).")
    args = ap.parse_args()

    apply_journal_style()

    path = (REPO_ROOT / str(args.summary)).resolve()
    if not path.exists():
        raise SystemExit(f"missing benchmark summary: {path}")

    df = pd.read_csv(path, sep="\t")
    req = {
        "context",
        "horizon_weeks",
        "pathogen",
        "mean_rmse_naive_persistence",
        "mean_rmse_ridge_ar",
        "mean_rmse_ridge_fusion",
        "mean_rmse_ridge_ww_only",
        "frac_fusion_better_than_ar",
    }
    miss = sorted(req - set(df.columns))
    if miss:
        raise SystemExit(f"benchmark summary missing columns: {', '.join(miss)}")

    contexts = [c for c in ["early_warning", "with_current_y"] if (df["context"].astype(str) == c).any()]
    pathogens = [p for p in ["COVID-19", "Influenza", "RSV"] if (df["pathogen"].astype(str) == p).any()]
    horizons = sorted(pd.to_numeric(df["horizon_weeks"], errors="coerce").dropna().astype(int).unique().tolist())

    df = df.copy()
    df["horizon_weeks"] = pd.to_numeric(df["horizon_weeks"], errors="coerce").astype(int)
    long = []
    for _, r in df.iterrows():
        for model_id, col in [
            ("naive_persistence", "mean_rmse_naive_persistence"),
            ("ridge_ar", "mean_rmse_ridge_ar"),
            ("ridge_ww_only", "mean_rmse_ridge_ww_only"),
            ("ridge_fusion", "mean_rmse_ridge_fusion"),
        ]:
            long.append(
                {
                    "context": str(r["context"]),
                    "pathogen": str(r["pathogen"]),
                    "horizon_weeks": int(r["horizon_weeks"]),
                    "model_id": model_id,
                    "mean_rmse": float(r[col]),
                    "frac_fusion_better_than_ar": float(r["frac_fusion_better_than_ar"]),
                }
            )
    dfl = pd.DataFrame(long)

    colors = {
        "naive_persistence": OKABE_ITO["gray"],
        "ridge_ar": OKABE_ITO["black"],
        "ridge_fusion": OKABE_ITO["vermillion"],
    }
    labels = {
        "naive_persistence": "Naive (persistence)",
        "ridge_ar": "Admissions-only",
        "ridge_fusion": "Fusion (admissions + wastewater)",
    }
    # Keep the main story clean: admissions-only vs fusion, with a naive comparator.
    model_order = ["naive_persistence", "ridge_ar", "ridge_fusion"]
    linestyles = {"naive_persistence": (0, (2, 2)), "ridge_ar": "solid", "ridge_fusion": "solid"}
    markers = {"naive_persistence": "o", "ridge_ar": "o", "ridge_fusion": "D"}

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    size = nature_double_column(height_in=5.1)
    fig = plt.figure(figsize=(size.width_in, size.height_in), constrained_layout=True)
    set_figure_facecolor(fig)
    gs = fig.add_gridspec(len(contexts), len(pathogens), wspace=0.25, hspace=0.35)

    for i, ctx in enumerate(contexts):
        for j, pathogen in enumerate(pathogens):
            ax = fig.add_subplot(gs[i, j])
            sub = dfl[(dfl["context"] == ctx) & (dfl["pathogen"] == pathogen)].copy()
            for mid in model_order:
                s = sub[sub["model_id"] == mid].sort_values("horizon_weeks")
                if s.empty:
                    continue
                ax.plot(
                    s["horizon_weeks"],
                    s["mean_rmse"],
                    marker=markers[mid],
                    markersize=3.2,
                    linewidth=1.6,
                    linestyle=linestyles[mid],
                    color=colors[mid],
                    label=labels[mid],
                )

            ctx_label = "Early warning" if ctx == "early_warning" else ("Nowcasting" if ctx == "with_current_y" else ctx)
            ax.set_title(f"{pathogen} ({ctx_label})", fontsize=9)
            ax.set_xticks(horizons)
            ax.set_xlabel("Lead time (weeks)")
            ax.set_ylabel("RMSE")
            ax.grid(True, axis="y")
            despine(ax)

            # The one-week fusion win rate is reported in the manuscript caption; we avoid in-panel callouts to keep the figure clean.

            if i == 0 and j == 0:
                ax.legend(loc="upper left", ncol=1, handlelength=2.2, columnspacing=1.2)

            panel_label(ax, chr(ord("A") + i * len(pathogens) + j))

    out_png, out_pdf = save_figure(fig, out_prefix=str(out_prefix), dpi=600)
    plt.close(fig)
    print(f"Wrote figure: {out_png}")
    print(f"Wrote figure: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
