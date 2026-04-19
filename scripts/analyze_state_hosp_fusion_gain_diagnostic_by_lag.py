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
        description="Create a lead–lag diagnostic table linking state-level best-lag association with per-series fusion gain (baseline cell; per100k)."
    )
    ap.add_argument(
        "--benchmark",
        default="results/stress/state_hosp_stress_grid/per100k_v1/state_hosp_forecast_benchmark.delay0_miss000_rev000.tsv",
        help="Benchmark TSV for the baseline cell (default: results/stress/state_hosp_stress_grid/per100k_v1/state_hosp_forecast_benchmark.delay0_miss000_rev000.tsv).",
    )
    ap.add_argument(
        "--leadlag",
        default="results/leadlag/wastewater_hosp_leadlag.geo_matched_state.tsv",
        help="Lead-lag association TSV (default: results/leadlag/wastewater_hosp_leadlag.geo_matched_state.tsv).",
    )
    ap.add_argument(
        "--dm",
        default="results/metrics/state_hosp_dm_test.per100k.tsv",
        help="Diebold–Mariano per-series tests (default: results/metrics/state_hosp_dm_test.per100k.tsv).",
    )
    ap.add_argument(
        "--svi-county-csv",
        default="data/raw/svi/2026-04-15_svi2022county01/svi_2022_county.csv",
        help="CDC/ATSDR SVI 2022 county CSV (default: data/raw/svi/2026-04-15_svi2022county01/svi_2022_county.csv).",
    )
    ap.add_argument(
        "--out",
        default="results/diagnostics/state_hosp_fusion_gain_diagnostic_by_lag.per100k.tsv",
        help="Output TSV (default: results/diagnostics/state_hosp_fusion_gain_diagnostic_by_lag.per100k.tsv).",
    )
    args = ap.parse_args()

    bench_path = (REPO_ROOT / str(args.benchmark)).resolve()
    lag_path = (REPO_ROOT / str(args.leadlag)).resolve()
    dm_path = (REPO_ROOT / str(args.dm)).resolve()
    svi_path = (REPO_ROOT / str(args.svi_county_csv)).resolve()
    for p in [bench_path, lag_path, dm_path, svi_path]:
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

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
    wide = wide.dropna(subset=["rmse_ar", "rmse_fusion"]).copy()
    wide["delta_rmse_fusion_minus_ar"] = wide["rmse_fusion"] - wide["rmse_ar"]
    wide["fusion_better_than_ar"] = (wide["delta_rmse_fusion_minus_ar"] < 0).astype(int)
    wide["relative_rmse_fusion_over_ar"] = wide["rmse_fusion"] / wide["rmse_ar"]
    wide["skill_score_fusion_vs_ar"] = 1.0 - wide["relative_rmse_fusion_over_ar"]

    lag = pd.read_csv(lag_path, sep="\t")
    needed_lag = {"geo_level", "geo_id", "pathogen", "best_lag_weeks", "corr_at_best", "n_pairs_at_best"}
    miss = sorted(needed_lag - set(lag.columns))
    if miss:
        raise SystemExit(f"leadlag missing columns: {', '.join(miss)}")
    lag = lag[lag["geo_level"] == "state"].copy()
    lag["best_lag_weeks"] = pd.to_numeric(lag["best_lag_weeks"], errors="coerce")
    lag["corr_at_best"] = pd.to_numeric(lag["corr_at_best"], errors="coerce")
    lag["n_pairs_at_best"] = pd.to_numeric(lag["n_pairs_at_best"], errors="coerce")

    dm = pd.read_csv(dm_path, sep="\t")
    needed_dm = {"geo_level", "geo_id", "pathogen", "horizon_weeks", "dm_p_value_two_sided", "mean_loss_diff_fusion_minus_ar"}
    miss = sorted(needed_dm - set(dm.columns))
    if miss:
        raise SystemExit(f"dm missing columns: {', '.join(miss)}")
    dm = dm[dm["geo_level"] == "state"].copy()
    dm["horizon_weeks"] = pd.to_numeric(dm["horizon_weeks"], errors="coerce")
    dm["dm_p_value_two_sided"] = pd.to_numeric(dm["dm_p_value_two_sided"], errors="coerce")
    dm["mean_loss_diff_fusion_minus_ar"] = pd.to_numeric(dm["mean_loss_diff_fusion_minus_ar"], errors="coerce")

    svi = pd.read_csv(svi_path)
    needed_svi = {"ST_ABBR", "RPL_THEMES", "E_TOTPOP"}
    miss = sorted(needed_svi - set(svi.columns))
    if miss:
        raise SystemExit(f"svi county CSV missing columns: {', '.join(miss)}")
    state_svi = (
        svi.groupby("ST_ABBR", as_index=False)
        .apply(lambda g: pd.Series({"svi_rpl_themes_popwt": _weighted_mean(g["RPL_THEMES"], g["E_TOTPOP"])}))
        .reset_index(drop=True)
    )
    state_svi["svi_tertile"] = _assign_tertiles(state_svi["svi_rpl_themes_popwt"])

    out = wide.merge(lag[["geo_id", "pathogen", "best_lag_weeks", "corr_at_best", "n_pairs_at_best"]], on=["geo_id", "pathogen"], how="left")
    out = out.merge(
        dm[["geo_id", "pathogen", "horizon_weeks", "dm_p_value_two_sided", "mean_loss_diff_fusion_minus_ar"]],
        on=["geo_id", "pathogen", "horizon_weeks"],
        how="left",
    )
    out = out.merge(state_svi.rename(columns={"ST_ABBR": "geo_id"}), on="geo_id", how="left")
    out["svi_tertile"] = out["svi_tertile"].fillna("unknown")

    out_path = (REPO_ROOT / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["context", "horizon_weeks", "geo_id", "pathogen"]).to_csv(out_path, sep="\t", index=False)

    print(f"OK fusion_gain_diag: wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
