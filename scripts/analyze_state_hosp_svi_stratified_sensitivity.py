#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    m = values.notna() & weights.notna() & (weights > 0)
    if not bool(m.any()):
        return float("nan")
    return float(np.average(values[m].astype(float), weights=weights[m].astype(float)))


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
        description="Exploratory SVI-stratified sensitivity summary for Route A (state-scale admissions benchmark; delay=0, miss=0, rev=0)."
    )
    ap.add_argument(
        "--svi-county-csv",
        default="data/raw/svi/2026-04-15_svi2022county01/svi_2022_county.csv",
        help="CDC/ATSDR SVI 2022 county CSV (default: data/raw/svi/2026-04-15_svi2022county01/svi_2022_county.csv).",
    )
    ap.add_argument(
        "--benchmark",
        default="results/stress/state_hosp_stress_grid/per100k_v1/state_hosp_forecast_benchmark.delay0_miss000_rev000.tsv",
        help="Benchmark TSV for the baseline cell (default: results/stress/state_hosp_stress_grid/per100k_v1/state_hosp_forecast_benchmark.delay0_miss000_rev000.tsv).",
    )
    ap.add_argument(
        "--out-by-series",
        default="results/metrics/state_hosp_svi_stratified_by_series.per100k.tsv",
        help="Output TSV with per-series deltas and SVI stratum labels.",
    )
    ap.add_argument(
        "--out-summary",
        default="results/metrics/state_hosp_svi_stratified_summary.per100k.tsv",
        help="Output TSV with stratum-level summaries.",
    )
    args = ap.parse_args()

    svi_path = (REPO_ROOT / str(args.svi_county_csv)).resolve()
    if not svi_path.exists():
        raise SystemExit(f"missing SVI county CSV: {svi_path}")

    df_svi = pd.read_csv(svi_path)
    needed_svi = {"ST_ABBR", "RPL_THEMES", "E_TOTPOP"}
    miss = sorted(needed_svi - set(df_svi.columns))
    if miss:
        raise SystemExit(f"SVI county CSV missing columns: {', '.join(miss)}")

    state_svi = (
        df_svi.groupby("ST_ABBR", as_index=False)
        .apply(lambda g: pd.Series({"svi_rpl_themes_popwt": _weighted_mean(g["RPL_THEMES"], g["E_TOTPOP"])}))
        .reset_index(drop=True)
    )
    state_svi["svi_tertile"] = _assign_tertiles(state_svi["svi_rpl_themes_popwt"])

    bench_path = (REPO_ROOT / str(args.benchmark)).resolve()
    if not bench_path.exists():
        raise SystemExit(f"missing benchmark: {bench_path}")
    df = pd.read_csv(bench_path, sep="\t")

    needed = {"subset", "context", "geo_level", "geo_id", "pathogen", "horizon_weeks", "model_id", "rmse", "n_test"}
    miss = sorted(needed - set(df.columns))
    if miss:
        raise SystemExit(f"benchmark missing columns: {', '.join(miss)}")

    df = df[df["subset"] == "geo_matched_state"].copy()
    df = df[df["geo_level"] == "state"].copy()
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

    wide = wide.merge(state_svi.rename(columns={"ST_ABBR": "geo_id"}), on="geo_id", how="left")
    wide["svi_tertile"] = wide["svi_tertile"].fillna("unknown")

    out_by_series = (REPO_ROOT / str(args.out_by_series)).resolve()
    out_by_series.parent.mkdir(parents=True, exist_ok=True)
    wide.sort_values(["context", "horizon_weeks", "geo_id", "pathogen"]).to_csv(out_by_series, sep="\t", index=False)

    summary = (
        wide.groupby(["context", "horizon_weeks", "svi_tertile"], as_index=False)
        .agg(
            n_series=("delta_rmse_fusion_minus_ar", "size"),
            n_unique_geos=("geo_id", "nunique"),
            mean_delta_rmse=("delta_rmse_fusion_minus_ar", "mean"),
            median_delta_rmse=("delta_rmse_fusion_minus_ar", "median"),
            mean_rmse_ar=("rmse_ar", "mean"),
            mean_rmse_fusion=("rmse_fusion", "mean"),
            win_rate=("fusion_better_than_ar", "mean"),
        )
        .sort_values(["context", "horizon_weeks", "svi_tertile"])
        .reset_index(drop=True)
    )
    summary["win_rate"] = summary["win_rate"].astype(float)

    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, sep="\t", index=False)

    print(f"OK svi_stratified: wrote {out_by_series.relative_to(REPO_ROOT)} and {out_summary.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
