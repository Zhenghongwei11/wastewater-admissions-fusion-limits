from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stratum:
    context: str
    horizon_weeks: int
    pathogen: str

    @property
    def label(self) -> str:
        return f"{self.context} | h={self.horizon_weeks} | {self.pathogen}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Make Figure 6 (Route A): temporal robustness across historical as-of cutoffs (state admissions target).")
    ap.add_argument(
        "--asof-summary",
        default="results/benchmark/state_hosp_asof_forecast_benchmark_summary.tsv",
        help="As-of benchmark summary TSV.",
    )
    ap.add_argument(
        "--out-prefix",
        default="plots/fig6_state_hosp_asof_stability_heatmap",
        help="Output prefix (no extension).",
    )
    ap.add_argument(
        "--out-matrix",
        default="results/benchmark/state_hosp_asof_stability_matrix.tsv",
        help="Derived long table (per stratum × as-of).",
    )
    ap.add_argument(
        "--out-overall",
        default="results/benchmark/state_hosp_asof_stability_overall.tsv",
        help="Derived overall trend table (per context × as-of).",
    )
    args = ap.parse_args()

    src = (REPO_ROOT / str(args.asof_summary)).resolve()
    if not src.exists():
        raise SystemExit(f"missing as-of summary: {src}")

    df = pd.read_csv(src, sep="\t")
    req = {
        "as_of_week_end",
        "context",
        "horizon_weeks",
        "pathogen",
        "n_series",
        "mean_rmse_ridge_ar",
        "mean_rmse_ridge_fusion",
        "frac_fusion_better_than_ar",
    }
    missing = sorted(req - set(df.columns))
    if missing:
        raise SystemExit(f"as-of summary missing columns: {', '.join(missing)}")

    if "subset" in df.columns:
        df = df[df["subset"].astype(str) == "geo_matched_state"].copy()

    df["as_of_week_end"] = df["as_of_week_end"].astype(str)
    df["context"] = df["context"].astype(str)
    df["pathogen"] = df["pathogen"].astype(str)
    df["horizon_weeks"] = pd.to_numeric(df["horizon_weeks"], errors="coerce").astype(int)
    df["n_series"] = pd.to_numeric(df["n_series"], errors="coerce").fillna(0).astype(int)
    df["mean_rmse_ridge_ar"] = pd.to_numeric(df["mean_rmse_ridge_ar"], errors="coerce")
    df["mean_rmse_ridge_fusion"] = pd.to_numeric(df["mean_rmse_ridge_fusion"], errors="coerce")
    df["frac_fusion_better_than_ar"] = pd.to_numeric(df["frac_fusion_better_than_ar"], errors="coerce")

    df["delta_rmse_fusion_minus_ar"] = df["mean_rmse_ridge_fusion"] - df["mean_rmse_ridge_ar"]

    out_rows = df[
        [
            "as_of_week_end",
            "context",
            "horizon_weeks",
            "pathogen",
            "n_series",
            "mean_rmse_ridge_ar",
            "mean_rmse_ridge_fusion",
            "delta_rmse_fusion_minus_ar",
            "frac_fusion_better_than_ar",
        ]
    ].copy()
    out_matrix_path = (REPO_ROOT / str(args.out_matrix)).resolve()
    out_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    out_rows.to_csv(out_matrix_path, sep="\t", index=False)

    overall = (
        df.groupby(["as_of_week_end", "context"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_strata": int(len(g)),
                    "mean_frac_fusion_better_than_ar": float(g["frac_fusion_better_than_ar"].mean()) if len(g) else math.nan,
                    "mean_delta_rmse_fusion_minus_ar": float(g["delta_rmse_fusion_minus_ar"].mean()) if len(g) else math.nan,
                    "weighted_delta_rmse_fusion_minus_ar": float((g["delta_rmse_fusion_minus_ar"] * g["n_series"]).sum() / max(int(g["n_series"].sum()), 1)),
                }
            )
        )
        .reset_index()
        .sort_values(["context", "as_of_week_end"])
    )
    out_overall_path = (REPO_ROOT / str(args.out_overall)).resolve()
    out_overall_path.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out_overall_path, sep="\t", index=False)

    asofs = sorted(df["as_of_week_end"].unique().tolist())
    strata = []
    for ctx in ["with_current_y", "early_warning"]:
        for h in sorted(df["horizon_weeks"].unique().tolist()):
            for p in ["COVID-19", "Influenza", "RSV"]:
                if ((df["context"] == ctx) & (df["horizon_weeks"] == h) & (df["pathogen"] == p)).any():
                    strata.append(Stratum(context=ctx, horizon_weeks=int(h), pathogen=p))

    if not strata or not asofs:
        raise SystemExit("no strata/as-of values to plot")

    mat = []
    for st in strata:
        row = []
        for asof in asofs:
            sub = df[(df["as_of_week_end"] == asof) & (df["context"] == st.context) & (df["horizon_weeks"] == st.horizon_weeks) & (df["pathogen"] == st.pathogen)]
            row.append(float(sub["delta_rmse_fusion_minus_ar"].iloc[0]) if not sub.empty else math.nan)
        mat.append(row)

    finite = [abs(v) for r in mat for v in r if isinstance(v, float) and math.isfinite(v)]
    vmax = max(finite) if finite else 1e-6
    vmax = max(vmax, 1e-6)

    out_prefix = (REPO_ROOT / str(args.out_prefix)).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], width_ratios=[2.2, 1.0])

    ax_hm = fig.add_subplot(gs[0, 0])
    im = ax_hm.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_hm.set_title("Figure 6 (Route A). Temporal robustness (as-of cutoffs): ΔRMSE (fusion − AR) on admissions")
    ax_hm.set_xticks(range(len(asofs)))
    ax_hm.set_xticklabels(asofs, rotation=30, ha="right", fontsize=9)
    ax_hm.set_yticks(range(len(strata)))
    ax_hm.set_yticklabels([s.label for s in strata], fontsize=8)
    ax_hm.set_xlabel("As-of cutoff (week_end ≤ cutoff)")
    ax_hm.set_ylabel("Stratum (context × horizon × pathogen)")

    for i in range(len(strata)):
        for j in range(len(asofs)):
            v = mat[i][j]
            if not isinstance(v, float) or not math.isfinite(v):
                continue
            ax_hm.text(j, i, f"{v:+.2g}", ha="center", va="center", fontsize=7, color="black")

    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ΔRMSE (fusion − AR)  (negative = fusion improves)")

    ax_line = fig.add_subplot(gs[1, :])
    for ctx in ["with_current_y", "early_warning"]:
        sub = overall[overall["context"] == ctx].copy()
        if sub.empty:
            continue
        ax_line.plot(sub["as_of_week_end"], sub["mean_frac_fusion_better_than_ar"], marker="o", label=f"{ctx}: mean win-rate")
    ax_line.set_ylim(0.0, 1.0)
    ax_line.set_ylabel("Mean fraction fusion better than AR")
    ax_line.set_xlabel("As-of cutoff")
    ax_line.grid(True, alpha=0.3)
    ax_line.legend(loc="best", fontsize=9)

    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Wrote derived matrix: {out_matrix_path}")
    print(f"Wrote overall trend: {out_overall_path}")
    print(f"Wrote figure: {out_png}")
    print(f"Wrote figure: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

