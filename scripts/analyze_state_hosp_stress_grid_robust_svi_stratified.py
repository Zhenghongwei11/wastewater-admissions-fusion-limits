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


def _fmt_frac(x: float) -> str:
    return f"{int(round(float(x) * 100)):03d}"


def _benchmark_path(*, delay: int, miss: float, rev: float, folder: Path) -> Path:
    tag = f"delay{int(delay)}_miss{_fmt_frac(miss)}_rev{_fmt_frac(rev)}"
    return folder / f"state_hosp_forecast_benchmark.{tag}.tsv"


def _agg_over_strata(df_strata: pd.DataFrame) -> dict:
    df_strata = df_strata.copy()
    df_strata["n_series"] = pd.to_numeric(df_strata["n_series"], errors="coerce").fillna(0).astype(int)
    df_strata["mean_delta_rmse_fusion_minus_ar"] = pd.to_numeric(df_strata["mean_delta_rmse_fusion_minus_ar"], errors="coerce")
    df_strata["frac_fusion_better_than_ar"] = pd.to_numeric(df_strata["frac_fusion_better_than_ar"], errors="coerce")

    w = df_strata["n_series"].to_numpy(dtype=float)
    denom = float(w.sum()) if float(w.sum()) > 0 else float("nan")
    weighted_delta = float((df_strata["mean_delta_rmse_fusion_minus_ar"] * w).sum() / denom) if denom == denom else float("nan")
    weighted_win = float((df_strata["frac_fusion_better_than_ar"] * w).sum() / denom) if denom == denom else float("nan")

    return {
        "n_strata": int(len(df_strata)),
        "total_series": int(df_strata["n_series"].sum()),
        "mean_delta_rmse_fusion_minus_ar": float(df_strata["mean_delta_rmse_fusion_minus_ar"].mean()) if len(df_strata) else float("nan"),
        "weighted_delta_rmse_fusion_minus_ar": weighted_delta,
        "mean_frac_fusion_better_than_ar": float(df_strata["frac_fusion_better_than_ar"].mean()) if len(df_strata) else float("nan"),
        "weighted_frac_fusion_better_than_ar": weighted_win,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SVI-tertile stratified summaries for the globally robust Route A stress-grid cells (reviewer-safe; uses the same robust-cell definition as the main analysis)."
    )
    ap.add_argument(
        "--svi-county-csv",
        default="data/raw/svi/2026-04-15_svi2022county01/svi_2022_county.csv",
        help="CDC/ATSDR SVI 2022 county CSV (default: data/raw/svi/2026-04-15_svi2022county01/svi_2022_county.csv).",
    )
    ap.add_argument(
        "--robust-cells",
        default="results/stress/state_hosp_stress_grid_robust_cells.per100k.tsv",
        help="Globally robust cells TSV (default: results/stress/state_hosp_stress_grid_robust_cells.per100k.tsv).",
    )
    ap.add_argument(
        "--bench-folder",
        default="results/stress/state_hosp_stress_grid/per100k_v1",
        help="Folder holding per-cell benchmark TSVs (default: results/stress/state_hosp_stress_grid/per100k_v1).",
    )
    ap.add_argument(
        "--out-cells",
        default="results/metrics/state_hosp_stress_grid_robust_svi_stratified_cells.per100k.tsv",
        help="Output TSV of per-cell, per-SVI-tertile summaries.",
    )
    ap.add_argument(
        "--out-summary",
        default="results/metrics/state_hosp_stress_grid_robust_svi_stratified_summary.per100k.tsv",
        help="Output TSV of best/worst robust-cell summaries per context and SVI tertile.",
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

    robust_path = (REPO_ROOT / str(args.robust_cells)).resolve()
    if not robust_path.exists():
        raise SystemExit(f"missing robust cells: {robust_path}")
    robust = pd.read_csv(robust_path, sep="\t")
    needed_rob = {"y_delay_weeks", "feature_missing_frac", "feature_revision_frac", "context"}
    miss = sorted(needed_rob - set(robust.columns))
    if miss:
        raise SystemExit(f"robust cells TSV missing columns: {', '.join(miss)}")

    bench_folder = (REPO_ROOT / str(args.bench_folder)).resolve()
    if not bench_folder.exists():
        raise SystemExit(f"missing benchmark folder: {bench_folder}")

    cell_rows: list[dict] = []
    for _, cell in robust.iterrows():
        delay = int(cell["y_delay_weeks"])
        miss_frac = float(cell["feature_missing_frac"])
        rev_frac = float(cell["feature_revision_frac"])
        context = str(cell["context"])

        bench_path = _benchmark_path(delay=delay, miss=miss_frac, rev=rev_frac, folder=bench_folder)
        if not bench_path.exists():
            raise SystemExit(f"missing benchmark TSV for robust cell: {bench_path}")

        df = pd.read_csv(bench_path, sep="\t")
        needed = {"subset", "context", "geo_level", "geo_id", "pathogen", "horizon_weeks", "model_id", "rmse"}
        miss_cols = sorted(needed - set(df.columns))
        if miss_cols:
            raise SystemExit(f"benchmark missing columns ({bench_path.name}): {', '.join(miss_cols)}")

        df = df[df["subset"] == "geo_matched_state"].copy()
        df = df[df["geo_level"] == "state"].copy()
        df = df[df["context"] == context].copy()
        df = df[df["model_id"].isin(["ridge_ar", "ridge_fusion"])].copy()
        df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
        df["horizon_weeks"] = pd.to_numeric(df["horizon_weeks"], errors="coerce")

        df = df.merge(state_svi.rename(columns={"ST_ABBR": "geo_id"}), on="geo_id", how="left")
        df["svi_tertile"] = df["svi_tertile"].fillna("unknown")

        wide = (
            df.pivot_table(
                index=["context", "horizon_weeks", "pathogen", "geo_id", "svi_tertile"],
                columns="model_id",
                values="rmse",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"ridge_ar": "rmse_ar", "ridge_fusion": "rmse_fusion"})
        )
        wide = wide.dropna(subset=["rmse_ar", "rmse_fusion"]).copy()
        wide["fusion_better"] = (wide["rmse_fusion"] < wide["rmse_ar"]).astype(int)

        strata = (
            wide.groupby(["context", "horizon_weeks", "pathogen", "svi_tertile"], as_index=False)
            .agg(
                n_series=("geo_id", "nunique"),
                mean_rmse_ridge_ar=("rmse_ar", "mean"),
                mean_rmse_ridge_fusion=("rmse_fusion", "mean"),
                frac_fusion_better_than_ar=("fusion_better", "mean"),
            )
            .reset_index(drop=True)
        )
        strata["mean_delta_rmse_fusion_minus_ar"] = strata["mean_rmse_ridge_fusion"] - strata["mean_rmse_ridge_ar"]

        for tertile, g in strata.groupby("svi_tertile", dropna=False):
            rec = _agg_over_strata(g)
            cell_rows.append(
                {
                    "context": context,
                    "svi_tertile": str(tertile),
                    "y_delay_weeks": delay,
                    "feature_missing_frac": miss_frac,
                    "feature_revision_frac": rev_frac,
                    **rec,
                }
            )

    out_cells = (REPO_ROOT / str(args.out_cells)).resolve()
    out_cells.parent.mkdir(parents=True, exist_ok=True)
    df_cells = pd.DataFrame(cell_rows).sort_values(
        ["context", "svi_tertile", "y_delay_weeks", "feature_missing_frac", "feature_revision_frac"]
    )
    df_cells.to_csv(out_cells, sep="\t", index=False)

    # Summarize in a Table-2-like shape per context × SVI tertile.
    def _fmt_params(r: pd.Series) -> str:
        return f"delay={int(r['y_delay_weeks'])},miss={float(r['feature_missing_frac']):g},rev={float(r['feature_revision_frac']):g}"

    summary_rows: list[dict] = []
    for (context, tertile), g in df_cells.groupby(["context", "svi_tertile"], dropna=False):
        g = g.copy()
        best_delta_row = g.sort_values("weighted_delta_rmse_fusion_minus_ar", ascending=True).iloc[0]
        worst_delta_row = g.sort_values("weighted_delta_rmse_fusion_minus_ar", ascending=False).iloc[0]
        best_win_row = g.sort_values("weighted_frac_fusion_better_than_ar", ascending=False).iloc[0]
        any_help = bool((g["weighted_delta_rmse_fusion_minus_ar"] < 0).any())
        summary_rows.append(
            {
                "context": str(context),
                "svi_tertile": str(tertile),
                "n_cells_robust_global": int(len(g)),
                "best_weighted_delta_rmse_fusion_minus_ar": float(best_delta_row["weighted_delta_rmse_fusion_minus_ar"]),
                "best_delta_params": _fmt_params(best_delta_row),
                "worst_weighted_delta_rmse_fusion_minus_ar": float(worst_delta_row["weighted_delta_rmse_fusion_minus_ar"]),
                "worst_delta_params": _fmt_params(worst_delta_row),
                "best_weighted_winrate_fusion_better": float(best_win_row["weighted_frac_fusion_better_than_ar"]),
                "best_winrate_params": _fmt_params(best_win_row),
                "any_cell_fusion_improves_weighted_rmse": int(any_help),
            }
        )

    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).sort_values(["context", "svi_tertile"]).to_csv(out_summary, sep="\t", index=False)

    print(f"OK robust_svi: wrote {out_cells.relative_to(REPO_ROOT)} and {out_summary.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

