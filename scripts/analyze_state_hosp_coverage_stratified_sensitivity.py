#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assign_tertiles(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    qs = s.quantile([1 / 3, 2 / 3], interpolation="linear")
    q1 = float(qs.iloc[0])
    q2 = float(qs.iloc[1])

    def _lab(x: float) -> str:
        if not np.isfinite(x):
            return "unknown"
        if x <= q1:
            return "low"
        if x <= q2:
            return "mid"
        return "high"

    return s.apply(_lab)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Exploratory coverage-stratified sensitivity summary for Route A (state-scale admissions benchmark; delay=0, miss=0, rev=0)."
    )
    ap.add_argument(
        "--panel",
        default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv",
        help="Geo-matched state panel TSV (default: results/panels/wastewater_hosp_panel.geo_matched_state.tsv).",
    )
    ap.add_argument(
        "--benchmark",
        default="results/stress/state_hosp_stress_grid/per100k_v1/state_hosp_forecast_benchmark.delay0_miss000_rev000.tsv",
        help="Benchmark TSV for the baseline cell (default: results/stress/state_hosp_stress_grid/per100k_v1/state_hosp_forecast_benchmark.delay0_miss000_rev000.tsv).",
    )
    ap.add_argument(
        "--out-by-series",
        default="results/metrics/state_hosp_coverage_stratified_by_series.per100k.tsv",
        help="Output TSV with per-series deltas and coverage tertile labels.",
    )
    ap.add_argument(
        "--out-summary",
        default="results/metrics/state_hosp_coverage_stratified_summary.per100k.tsv",
        help="Output TSV with coverage-tertile summaries.",
    )
    args = ap.parse_args()

    panel_path = (REPO_ROOT / str(args.panel)).resolve()
    bench_path = (REPO_ROOT / str(args.benchmark)).resolve()
    for p in [panel_path, bench_path]:
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    panel = pd.read_csv(panel_path, sep="\t", dtype={"geo_id": str, "pathogen": str, "subset": str})
    if "nwss_n_samples_sum" not in panel.columns:
        raise SystemExit("panel missing nwss_n_samples_sum")
    if "nwss_population_coverage_max" not in panel.columns:
        raise SystemExit("panel missing nwss_population_coverage_max")
    panel["nwss_n_samples_sum"] = pd.to_numeric(panel["nwss_n_samples_sum"], errors="coerce").fillna(0.0)
    panel["nwss_population_coverage_max"] = pd.to_numeric(panel["nwss_population_coverage_max"], errors="coerce")
    panel = panel[panel["subset"].astype(str) == "geo_matched_state"].copy()
    cov = (
        panel.groupby(["geo_id", "pathogen"], as_index=False)
        .agg(
            n_weeks=("week_end", "size"),
            median_samples_per_week=("nwss_n_samples_sum", "median"),
            mean_samples_per_week=("nwss_n_samples_sum", "mean"),
            p10_samples=("nwss_n_samples_sum", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.1))),
            p90_samples=("nwss_n_samples_sum", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.9))),
            median_population_coverage=("nwss_population_coverage_max", "median"),
            mean_population_coverage=("nwss_population_coverage_max", "mean"),
        )
        .reset_index(drop=True)
    )
    cov["coverage_tertile_median_population_coverage"] = _assign_tertiles(cov["median_population_coverage"])
    cov["coverage_tertile_median_samples"] = _assign_tertiles(cov["median_samples_per_week"])

    df = pd.read_csv(bench_path, sep="\t")
    needed = {"subset", "context", "geo_level", "geo_id", "pathogen", "horizon_weeks", "model_id", "rmse", "n_test"}
    miss = sorted(needed - set(df.columns))
    if miss:
        raise SystemExit(f"benchmark missing columns: {', '.join(miss)}")

    df = df[(df["subset"] == "geo_matched_state") & (df["geo_level"] == "state")].copy()
    df = df[df["model_id"].isin(["ridge_ar", "ridge_fusion"])].copy()
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["horizon_weeks"] = pd.to_numeric(df["horizon_weeks"], errors="coerce")
    df["test_weeks"] = pd.to_numeric(df["n_test"], errors="coerce")

    wide = (
        df.pivot_table(
            index=["context", "geo_id", "pathogen", "horizon_weeks", "test_weeks"],
            columns="model_id",
            values="rmse",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"ridge_ar": "rmse_ar", "ridge_fusion": "rmse_fusion"})
    )
    wide["delta_rmse_fusion_minus_ar"] = wide["rmse_fusion"] - wide["rmse_ar"]
    wide["fusion_better_than_ar"] = (wide["delta_rmse_fusion_minus_ar"] < 0).astype(int)

    wide = wide.merge(cov, on=["geo_id", "pathogen"], how="left")
    wide["coverage_tertile_median_population_coverage"] = wide["coverage_tertile_median_population_coverage"].fillna("unknown")
    wide["coverage_tertile_median_samples"] = wide["coverage_tertile_median_samples"].fillna("unknown")

    out_by_series = (REPO_ROOT / str(args.out_by_series)).resolve()
    out_by_series.parent.mkdir(parents=True, exist_ok=True)
    wide.sort_values(["context", "horizon_weeks", "geo_id", "pathogen"]).to_csv(out_by_series, sep="\t", index=False)

    summary = (
        wide.groupby(["context", "horizon_weeks", "coverage_tertile_median_population_coverage"], as_index=False)
        .agg(
            n_series=("delta_rmse_fusion_minus_ar", "size"),
            n_unique_geos=("geo_id", "nunique"),
            mean_delta_rmse=("delta_rmse_fusion_minus_ar", "mean"),
            median_delta_rmse=("delta_rmse_fusion_minus_ar", "median"),
            mean_rmse_ar=("rmse_ar", "mean"),
            mean_rmse_fusion=("rmse_fusion", "mean"),
            win_rate=("fusion_better_than_ar", "mean"),
            median_samples_per_week=("median_samples_per_week", "median"),
            median_population_coverage=("median_population_coverage", "median"),
        )
        .sort_values(["context", "horizon_weeks", "coverage_tertile_median_population_coverage"])
        .reset_index(drop=True)
    )
    summary["win_rate"] = summary["win_rate"].astype(float)

    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, sep="\t", index=False)

    print(f"OK coverage_stratified: wrote {out_by_series.relative_to(REPO_ROOT)} and {out_summary.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
