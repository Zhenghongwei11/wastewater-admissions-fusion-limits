from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Threshold:
    min_strata: int
    min_total_series: int


def main() -> int:
    ap = argparse.ArgumentParser(description="Sensitivity scan for Route A stress-grid robust-scenario thresholds (no grid re-run).")
    ap.add_argument(
        "--overall",
        default="results/stress/state_hosp_stress_grid_overall.per100k.tsv",
        help="Stress grid overall TSV (default: results/stress/state_hosp_stress_grid_overall.per100k.tsv).",
    )
    ap.add_argument("--min-strata", type=int, default=12, help="Minimum strata coverage threshold (default: 12).")
    ap.add_argument(
        "--min-total-series-list",
        default="200,300,400",
        help="Comma-separated thresholds for minimum total series (default: 200,300,400).",
    )
    ap.add_argument(
        "--out",
        default="results/stress/state_hosp_stress_grid_robust_threshold_sensitivity.per100k.tsv",
        help="Output TSV path.",
    )
    args = ap.parse_args()

    overall = (REPO_ROOT / str(args.overall)).resolve()
    df = pd.read_csv(overall, sep="\t")

    required = [
        "context",
        "n_strata",
        "total_series",
        "weighted_delta_rmse_fusion_minus_ar",
        "weighted_frac_fusion_better_than_ar",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"overall table missing columns: {missing}")

    df = df.copy()
    df["n_strata"] = pd.to_numeric(df["n_strata"], errors="coerce")
    df["total_series"] = pd.to_numeric(df["total_series"], errors="coerce")
    df["weighted_delta_rmse_fusion_minus_ar"] = pd.to_numeric(df["weighted_delta_rmse_fusion_minus_ar"], errors="coerce")
    df["weighted_frac_fusion_better_than_ar"] = pd.to_numeric(df["weighted_frac_fusion_better_than_ar"], errors="coerce")

    min_series_vals = [int(x.strip()) for x in str(args.min_total_series_list).split(",") if x.strip()]
    min_series_vals = sorted(set(min_series_vals))
    if not min_series_vals:
        raise SystemExit("--min-total-series-list must contain at least one integer")

    rows: list[dict] = []
    for min_total_series in min_series_vals:
        thr = Threshold(min_strata=int(args.min_strata), min_total_series=int(min_total_series))
        robust = df[(df["n_strata"] >= thr.min_strata) & (df["total_series"] >= thr.min_total_series)].copy()

        for context, g_all in df.groupby("context"):
            g_rob = robust[robust["context"] == context].copy()

            baseline = g_all[
                (g_all["y_delay_weeks"] == 0)
                & (g_all["feature_missing_frac"] == 0.0)
                & (g_all["feature_revision_frac"] == 0.0)
            ]
            baseline_row = baseline.iloc[0] if len(baseline) else None

            best_delta = float(g_rob["weighted_delta_rmse_fusion_minus_ar"].min()) if len(g_rob) else float("nan")
            worst_delta = float(g_rob["weighted_delta_rmse_fusion_minus_ar"].max()) if len(g_rob) else float("nan")
            best_win = float(g_rob["weighted_frac_fusion_better_than_ar"].max()) if len(g_rob) else float("nan")
            any_help = bool((g_rob["weighted_delta_rmse_fusion_minus_ar"] < 0).any()) if len(g_rob) else False

            rows.append(
                {
                    "context": context,
                    "min_strata": thr.min_strata,
                    "min_total_series": thr.min_total_series,
                    "n_cells_total": int(len(g_all)),
                    "n_cells_robust": int(len(g_rob)),
                    "baseline_weighted_delta": float(baseline_row["weighted_delta_rmse_fusion_minus_ar"]) if baseline_row is not None else float("nan"),
                    "baseline_weighted_winrate": float(baseline_row["weighted_frac_fusion_better_than_ar"]) if baseline_row is not None else float("nan"),
                    "robust_best_weighted_delta": best_delta,
                    "robust_worst_weighted_delta": worst_delta,
                    "robust_best_weighted_winrate": best_win,
                    "robust_any_cell_fusion_improves_rmse": int(any_help),
                }
            )

    out = (REPO_ROOT / str(args.out)).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["context", "min_total_series"]).to_csv(out, sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

