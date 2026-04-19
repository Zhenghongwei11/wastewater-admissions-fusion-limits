from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkConfig:
    panel_tsv: str
    subset: str
    y_lags: int
    ww_lags: int
    test_weeks: int
    ridge_alpha: float
    mape_epsilon: float
    include_seasonal_naive: bool
    context: str  # with_current_y | early_warning | both
    horizons: list[int]
    y_delay_weeks: int  # reporting delay for admissions features (0 = current behavior)
    feature_missing_frac: float  # synthetic missingness applied to admissions features only
    feature_revision_frac: float  # synthetic revision probability applied to admissions features only
    feature_revision_scale: float  # multiplicative scale (e.g., 0.2 => +/-20%)
    feature_seed: int  # deterministic seed for synthetic corruption


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a geo-matched state-level forecasting benchmark (NHSN hospital admissions target) and signal ablation.")
    ap.add_argument(
        "--panel",
        default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv",
        help="Joined panel TSV (wastewater × hospital admissions).",
    )
    ap.add_argument("--subset", default="geo_matched_state", help="Subset label to filter (default: geo_matched_state).")
    ap.add_argument(
        "--y-col",
        default="hosp_admissions",
        help="Target column in the panel (default: hosp_admissions). Suggested: hosp_admissions_per_100k, hosp_admissions_log1p, hosp_admissions_per_100k_log1p.",
    )
    ap.add_argument(
        "--y-delay-weeks",
        type=int,
        default=0,
        help="Synthetic reporting delay in weeks for admissions features only. A delay of d excludes y_lag0..y_lag(d-1) from AR/fusion and uses y_lag(d) for naive persistence (default: 0).",
    )
    ap.add_argument(
        "--feature-missing-frac",
        type=float,
        default=0.0,
        help="Synthetic missing fraction applied to admissions features only (0..1; default: 0.0). Rows with missing features are dropped.",
    )
    ap.add_argument(
        "--feature-revision-frac",
        type=float,
        default=0.0,
        help="Synthetic revision probability applied to admissions features only (0..1; default: 0.0). Revised points are multiplicatively perturbed by +/-feature-revision-scale.",
    )
    ap.add_argument(
        "--feature-revision-scale",
        type=float,
        default=0.2,
        help="Synthetic revision multiplicative scale (default: 0.2 => +/-20 percent). Only used when feature-revision-frac > 0.",
    )
    ap.add_argument("--feature-seed", type=int, default=7, help="Seed for synthetic corruption (default: 7).")
    ap.add_argument("--horizon", type=int, default=1, help="Forecast horizon in weeks (default: 1). Ignored if --horizons is set.")
    ap.add_argument("--horizons", default=None, help="Optional comma-separated horizons (e.g., 1,2,3,4).")
    ap.add_argument("--y-lags", type=int, default=4, help="Number of admissions autoregressive lags (default: 4).")
    ap.add_argument("--ww-lags", type=int, default=4, help="Number of wastewater lags (default: 4).")
    ap.add_argument("--test-weeks", type=int, default=52, help="Holdout weeks per series (default: 52).")
    ap.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization alpha (default: 1.0).")
    ap.add_argument("--mape-eps", type=float, default=1e-5, help="Epsilon for MAPE stability (default: 1e-5).")
    ap.add_argument(
        "--include-seasonal-naive",
        action="store_true",
        help="Also evaluate a 52-week seasonal naive baseline when available.",
    )
    ap.add_argument(
        "--context",
        default="both",
        choices=["with_current_y", "early_warning", "both"],
        help="Forecast context. early_warning excludes contemporaneous y (y_lag0) from models and baselines. both emits both contexts.",
    )
    ap.add_argument(
        "--out-benchmark",
        default="results/benchmark/state_hosp_forecast_benchmark.tsv",
        help="Benchmark TSV output.",
    )
    ap.add_argument(
        "--out-summary",
        default="results/benchmark/state_hosp_forecast_benchmark_summary.tsv",
        help="Small summary TSV (means + fusion win-rate) output.",
    )
    ap.add_argument(
        "--out-predictions",
        default="",
        help="Optional long-format predictions TSV output (one row per series-week-model). When set, writes predicted and true values for the holdout weeks.",
    )
    ap.add_argument(
        "--out-ablation",
        default="results/ablation/state_hosp_signal_ablation.tsv",
        help="Ablation TSV output.",
    )
    args = ap.parse_args()

    horizons = [int(args.horizon)]
    if args.horizons:
        horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
        horizons = sorted({h for h in horizons if h >= 1})
        if not horizons:
            raise SystemExit("--horizons must include at least one integer >= 1")

    cfg = BenchmarkConfig(
        panel_tsv=str(args.panel),
        subset=str(args.subset),
        y_lags=int(args.y_lags),
        ww_lags=int(args.ww_lags),
        test_weeks=int(args.test_weeks),
        ridge_alpha=float(args.ridge_alpha),
        mape_epsilon=float(args.mape_eps),
        include_seasonal_naive=bool(args.include_seasonal_naive),
        context=str(args.context),
        horizons=horizons,
        y_delay_weeks=int(args.y_delay_weeks),
        feature_missing_frac=float(args.feature_missing_frac),
        feature_revision_frac=float(args.feature_revision_frac),
        feature_revision_scale=float(args.feature_revision_scale),
        feature_seed=int(args.feature_seed),
    )

    y_col = str(args.y_col).strip()
    df = pd.read_csv(REPO_ROOT / cfg.panel_tsv, sep="\t", dtype={"geo_id": str, "pathogen": str, "subset": str, "hosp_source_id": str})
    df = df.loc[df["subset"].astype(str) == cfg.subset].copy()
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    if y_col not in df.columns:
        raise SystemExit(f"--y-col not found in panel: {y_col}")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df["nwss_conc_mean"] = pd.to_numeric(df["nwss_conc_mean"], errors="coerce")
    df = df.dropna(subset=["week_end", y_col, "nwss_conc_mean"]).copy()
    df = df.sort_values(["geo_id", "pathogen", "week_end"])

    benchmark_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    contexts = ["with_current_y", "early_warning"] if cfg.context == "both" else [cfg.context]

    for (geo_id, pathogen), g in df.groupby(["geo_id", "pathogen"], sort=True):
        g = g.sort_values("week_end").reset_index(drop=True)
        g = g.copy()
        g["__y_target"] = pd.to_numeric(g[y_col], errors="coerce")
        g["__y_feat"] = _apply_synthetic_corruption(
            g["__y_target"].to_numpy(dtype=float),
            geo_id=str(geo_id),
            pathogen=str(pathogen),
            cfg=cfg,
        )
        source_id = str(g["hosp_source_id"].iloc[-1]) if "hosp_source_id" in g.columns and len(g) else ""
        for horizon in cfg.horizons:
            series = _build_supervised(g, cfg, horizon_weeks=horizon)
            if series is None:
                continue
            X_all, y_all, t_all, idx_target_all, meta, y_series = series

            # Keep a small floor, but allow earlier as-of cutoffs by tying it to test_weeks.
            if len(y_all) < max(cfg.test_weeks + 20, 40):
                continue

            X_train, y_train, X_test, y_test, t_test, idx_target_test = _train_test_split(
                X_all, y_all, t_all, idx_target_all, test_weeks=cfg.test_weeks
            )
            min_train = max(26, cfg.y_lags + cfg.ww_lags + 20)
            if len(y_test) < 13 or len(y_train) < min_train:
                continue

            for ctx in contexts:
                models = _define_models(cfg)
                preds: dict[str, np.ndarray] = {}

                # Pre-slice feature blocks (optionally exclude contemporaneous y).
                idx_y = meta["idx_y"]
                drop = int(max(cfg.y_delay_weeks, 0) + (1 if ctx == "early_warning" else 0))
                if drop >= cfg.y_lags:
                    continue
                if drop > 0:
                    idx_y = idx_y[drop:]
                idx_ww = meta["idx_ww"]
                idx_season = meta["idx_season"]

                X_train_ar = X_train[:, idx_y + idx_season]
                X_test_ar = X_test[:, idx_y + idx_season]

                X_train_ww = X_train[:, idx_ww + idx_season]
                X_test_ww = X_test[:, idx_ww + idx_season]

                X_train_fusion = X_train[:, idx_y + idx_ww + idx_season]
                X_test_fusion = X_test[:, idx_y + idx_ww + idx_season]

                # Enforce a common finite row set across models within this context.
                mask_train = np.isfinite(X_train_fusion).all(axis=1) & np.isfinite(y_train)
                mask_test = np.isfinite(X_test_fusion).all(axis=1) & np.isfinite(y_test)
                if mask_train.sum() < min_train or mask_test.sum() < 13:
                    continue

                X_train_ar_i = X_train_ar[mask_train]
                X_test_ar_i = X_test_ar[mask_test]
                X_train_ww_i = X_train_ww[mask_train]
                X_test_ww_i = X_test_ww[mask_test]
                X_train_fusion_i = X_train_fusion[mask_train]
                X_test_fusion_i = X_test_fusion[mask_test]
                y_train_i = y_train[mask_train]
                y_test_i = y_test[mask_test]
                t_test_i = t_test[mask_test]
                idx_target_test_i = idx_target_test[mask_test]

                for model_id, model in models.items():
                    if model_id == "naive_persistence":
                        preds[model_id] = X_test[mask_test, int(drop)]
                        continue
                    if model_id == "naive_seasonal_52":
                        if not cfg.include_seasonal_naive:
                            continue
                        y_seas = []
                        for j in idx_target_test_i:
                            k = int(j) - 52
                            if k < 0:
                                y_seas.append(np.nan)
                            else:
                                y_seas.append(float(y_series[k]))
                        arr = np.asarray(y_seas, dtype=float)
                        keep = np.isfinite(arr)
                        if keep.sum() < 20:
                            continue
                        preds[model_id] = arr
                        continue
                    if model_id == "ridge_ar":
                        model.fit(X_train_ar_i, y_train_i)
                        preds[model_id] = model.predict(X_test_ar_i)
                        continue
                    if model_id == "ridge_ww_only":
                        model.fit(X_train_ww_i, y_train_i)
                        preds[model_id] = model.predict(X_test_ww_i)
                        continue
                    if model_id == "ridge_fusion":
                        model.fit(X_train_fusion_i, y_train_i)
                        preds[model_id] = model.predict(X_test_fusion_i)
                        continue

                for model_id, y_hat in preds.items():
                    if model_id == "naive_seasonal_52":
                        keep = np.isfinite(y_hat)
                        m = _metrics(y_test_i[keep], y_hat[keep], eps=cfg.mape_epsilon)
                        n_test_eff = int(keep.sum())
                        t_start = t_test_i[keep].min()
                        t_end = t_test_i[keep].max()
                    else:
                        m = _metrics(y_test_i, y_hat, eps=cfg.mape_epsilon)
                        n_test_eff = int(len(y_test_i))
                        t_start = t_test_i.min()
                        t_end = t_test_i.max()

                    benchmark_rows.append(
                        {
                            "subset": cfg.subset,
                            "context": ctx,
                            "geo_level": "state",
                            "geo_id": geo_id,
                            "pathogen": pathogen,
                            "hosp_source_id": source_id,
                            "horizon_weeks": int(horizon),
                            "model_id": model_id,
                            "n_train": int(len(y_train_i)),
                            "n_test": n_test_eff,
                            **m,
                            "test_start_week_end": pd.Timestamp(t_start).date().isoformat(),
                            "test_end_week_end": pd.Timestamp(t_end).date().isoformat(),
                        }
                    )

                    if str(args.out_predictions).strip():
                        if model_id == "naive_seasonal_52":
                            keep_pred = np.isfinite(y_hat)
                            y_true_out = y_test_i[keep_pred]
                            y_hat_out = y_hat[keep_pred]
                            t_out = t_test_i[keep_pred]
                        else:
                            keep_pred = np.isfinite(y_hat) & np.isfinite(y_test_i)
                            y_true_out = y_test_i[keep_pred]
                            y_hat_out = y_hat[keep_pred]
                            t_out = t_test_i[keep_pred]

                        for tt, yy, yh in zip(t_out, y_true_out, y_hat_out, strict=True):
                            pred_rows.append(
                                {
                                    "subset": cfg.subset,
                                    "context": ctx,
                                    "geo_level": "state",
                                    "geo_id": geo_id,
                                    "pathogen": pathogen,
                                    "hosp_source_id": source_id,
                                    "horizon_weeks": int(horizon),
                                    "model_id": model_id,
                                    "week_end": pd.Timestamp(tt).date().isoformat(),
                                    "y_true": float(yy),
                                    "y_hat": float(yh),
                                }
                            )

                if "ridge_fusion" in preds and "ridge_ar" in preds:
                    m_f = _metrics(y_test_i, preds["ridge_fusion"], eps=cfg.mape_epsilon)
                    m_a = _metrics(y_test_i, preds["ridge_ar"], eps=cfg.mape_epsilon)
                    ablation_rows.append(
                        {
                            "subset": cfg.subset,
                            "context": ctx,
                            "geo_level": "state",
                            "geo_id": geo_id,
                            "pathogen": pathogen,
                            "hosp_source_id": source_id,
                            "horizon_weeks": int(horizon),
                            "comparison": "ridge_fusion_minus_ridge_ar",
                            "delta_rmse": float(m_f["rmse"] - m_a["rmse"]),
                            "delta_mae": float(m_f["mae"] - m_a["mae"]),
                            "delta_mape": float(m_f["mape"] - m_a["mape"]),
                            "interpretation": "Negative delta indicates fusion improves over AR baseline.",
                        }
                    )
                if "ridge_fusion" in preds and "ridge_ww_only" in preds:
                    m_f = _metrics(y_test_i, preds["ridge_fusion"], eps=cfg.mape_epsilon)
                    m_w = _metrics(y_test_i, preds["ridge_ww_only"], eps=cfg.mape_epsilon)
                    ablation_rows.append(
                        {
                            "subset": cfg.subset,
                            "context": ctx,
                            "geo_level": "state",
                            "geo_id": geo_id,
                            "pathogen": pathogen,
                            "hosp_source_id": source_id,
                            "horizon_weeks": int(horizon),
                            "comparison": "ridge_fusion_minus_ridge_ww_only",
                            "delta_rmse": float(m_f["rmse"] - m_w["rmse"]),
                            "delta_mae": float(m_f["mae"] - m_w["mae"]),
                            "delta_mape": float(m_f["mape"] - m_w["mape"]),
                            "interpretation": "Negative delta indicates adding AR structure improves over wastewater-only.",
                        }
                    )

    out_b = (REPO_ROOT / str(args.out_benchmark)).resolve()
    out_b.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_b,
        benchmark_rows,
        fieldnames=[
            "subset",
            "context",
            "geo_level",
            "geo_id",
            "pathogen",
            "hosp_source_id",
            "horizon_weeks",
            "model_id",
            "n_train",
            "n_test",
            "rmse",
            "mae",
            "mape",
            "test_start_week_end",
            "test_end_week_end",
        ],
    )

    out_a = (REPO_ROOT / str(args.out_ablation)).resolve()
    out_a.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_a,
        ablation_rows,
        fieldnames=[
            "subset",
            "context",
            "geo_level",
            "geo_id",
            "pathogen",
            "hosp_source_id",
            "horizon_weeks",
            "comparison",
            "delta_rmse",
            "delta_mae",
            "delta_mape",
            "interpretation",
        ],
    )

    out_s = (REPO_ROOT / str(args.out_summary)).resolve()
    out_s.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_s,
        _build_summary(benchmark_rows, ablation_rows),
        fieldnames=[
            "subset",
            "context",
            "horizon_weeks",
            "pathogen",
            "n_series",
            "mean_rmse_naive_persistence",
            "mean_rmse_ridge_ar",
            "mean_rmse_ridge_fusion",
            "mean_rmse_ridge_ww_only",
            "frac_fusion_better_than_ar",
        ],
    )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "target_y_col": y_col,
        "inputs": {"panel_tsv": cfg.panel_tsv},
        "outputs": {"benchmark_tsv": str(Path(args.out_benchmark)), "summary_tsv": str(Path(args.out_summary)), "ablation_tsv": str(Path(args.out_ablation))},
        "rows": {"benchmark": int(len(benchmark_rows)), "summary": int(sum(1 for _ in _build_summary(benchmark_rows, ablation_rows))), "ablation": int(len(ablation_rows))},
    }
    (out_b.with_suffix(out_b.suffix + ".meta.json")).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if str(args.out_predictions).strip():
        out_p = (REPO_ROOT / str(args.out_predictions)).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        _write_tsv(
            out_p,
            pred_rows,
            fieldnames=[
                "subset",
                "context",
                "geo_level",
                "geo_id",
                "pathogen",
                "hosp_source_id",
                "horizon_weeks",
                "model_id",
                "week_end",
                "y_true",
                "y_hat",
            ],
        )
        (out_p.with_suffix(out_p.suffix + ".meta.json")).write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": asdict(cfg),
                    "target_y_col": y_col,
                    "inputs": {"panel_tsv": cfg.panel_tsv},
                    "outputs": {"predictions_tsv": str(Path(args.out_predictions))},
                    "rows": {"predictions": int(len(pred_rows))},
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print(str(out_b))
    print(str(out_s))
    print(str(out_a))
    return 0


def _define_models(cfg: BenchmarkConfig) -> dict[str, Any]:
    def ridge() -> Pipeline:
        return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", Ridge(alpha=cfg.ridge_alpha))])

    return {
        "naive_persistence": None,
        "naive_seasonal_52": None,
        "ridge_ar": ridge(),
        "ridge_ww_only": ridge(),
        "ridge_fusion": ridge(),
    }


def _build_supervised(g: pd.DataFrame, cfg: BenchmarkConfig, *, horizon_weeks: int):
    # This runner supports multiple target columns; main() normalizes g to contain __y.
    y_series = g["__y_target"].to_numpy(dtype=float)
    y_target = g["__y_target"].to_numpy(dtype=float)
    y_feat = g["__y_feat"].to_numpy(dtype=float)
    ww = g["nwss_conc_mean"].to_numpy(dtype=float)
    dt = pd.to_datetime(g["week_end"])
    t = dt.to_numpy()

    ww_log = np.log10(1.0 + np.clip(ww, a_min=0.0, a_max=None))

    max_lag = max(cfg.y_lags - 1, cfg.ww_lags - 1)
    if len(y_target) <= max_lag + horizon_weeks:
        return None

    # Seasonality features computed at feature time i.
    week = dt.dt.isocalendar().week.astype(int).to_numpy()
    ang = 2.0 * math.pi * (week / 53.0)
    sin_w = np.sin(ang)
    cos_w = np.cos(ang)

    col_y_lag0 = 0
    col_y_lag1 = 1 if cfg.y_lags >= 2 else 0
    idx_y = list(range(0, cfg.y_lags))
    idx_ww = list(range(cfg.y_lags, cfg.y_lags + cfg.ww_lags))
    idx_season = [cfg.y_lags + cfg.ww_lags, cfg.y_lags + cfg.ww_lags + 1]

    rows = []
    targets = []
    times = []
    idx_target = []

    for i in range(max_lag, len(y_target) - horizon_weeks):
        feat = []
        for j in range(cfg.y_lags):
            feat.append(float(y_feat[i - j]))
        for j in range(cfg.ww_lags):
            feat.append(float(ww_log[i - j]))
        feat.append(float(sin_w[i]))
        feat.append(float(cos_w[i]))

        rows.append(feat)
        yt = float(y_target[i + horizon_weeks])
        if not math.isfinite(yt):
            continue
        targets.append(yt)
        times.append(pd.Timestamp(t[i + horizon_weeks]))
        idx_target.append(int(i + horizon_weeks))

    X = np.asarray(rows, dtype=float)
    y_next = np.asarray(targets, dtype=float)
    t_next = np.asarray(times)
    idx_target_next = np.asarray(idx_target, dtype=int)

    meta = {
        "col_y_lag0": int(col_y_lag0),
        "col_y_lag1": int(col_y_lag1),
        "idx_y": idx_y,
        "idx_ww": idx_ww,
        "idx_season": idx_season,
    }
    return X, y_next, t_next, idx_target_next, meta, y_series


def _apply_synthetic_corruption(y: np.ndarray, *, geo_id: str, pathogen: str, cfg: BenchmarkConfig) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = y.copy()
    if len(out) == 0:
        return out

    missing = float(cfg.feature_missing_frac)
    rev_p = float(cfg.feature_revision_frac)
    scale = float(cfg.feature_revision_scale)
    if missing <= 0.0 and rev_p <= 0.0:
        return out

    # Stable per-series RNG: seed + deterministic hash.
    h = (geo_id + "|" + pathogen).encode("utf-8", errors="ignore")
    series_seed = int(cfg.feature_seed) ^ (int(np.frombuffer(h, dtype=np.uint8).sum()) + 9973 * len(h))
    rng = np.random.default_rng(series_seed)

    valid = np.isfinite(out)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return out

    if rev_p > 0.0 and scale > 0.0:
        rev_mask = rng.random(idx.size) < min(max(rev_p, 0.0), 1.0)
        if rev_mask.any():
            eps = rng.uniform(low=-abs(scale), high=abs(scale), size=int(rev_mask.sum()))
            out[idx[rev_mask]] = np.maximum(out[idx[rev_mask]] * (1.0 + eps), 0.0)

    if missing > 0.0:
        miss_mask = rng.random(idx.size) < min(max(missing, 0.0), 1.0)
        if miss_mask.any():
            out[idx[miss_mask]] = np.nan

    return out


def _train_test_split(X: np.ndarray, y: np.ndarray, t: np.ndarray, idx_target: np.ndarray, *, test_weeks: int):
    if len(y) <= test_weeks + 1:
        return X, y, X[:0], y[:0], t[:0], idx_target[:0]
    split = len(y) - int(test_weeks)
    return X[:split], y[:split], X[split:], y[split:], t[split:], idx_target[split:]


def _metrics(y: np.ndarray, y_hat: np.ndarray, *, eps: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    err = y_hat - y
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y), eps)
    mape = float(np.mean(np.abs(err) / denom))
    return {"rmse": rmse, "mae": mae, "mape": mape}


def _write_tsv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_summary(bench_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not bench_rows:
        return []
    b = pd.DataFrame(bench_rows)
    b["rmse"] = pd.to_numeric(b["rmse"], errors="coerce")
    key = ["subset", "context", "horizon_weeks", "pathogen"]

    means = (
        b.pivot_table(index=key, columns="model_id", values="rmse", aggfunc="mean")
        .reset_index()
        .rename(
            columns={
                "naive_persistence": "mean_rmse_naive_persistence",
                "ridge_ar": "mean_rmse_ridge_ar",
                "ridge_fusion": "mean_rmse_ridge_fusion",
                "ridge_ww_only": "mean_rmse_ridge_ww_only",
            }
        )
    )
    n = b.groupby(key)["geo_id"].nunique().reset_index().rename(columns={"geo_id": "n_series"})
    out = means.merge(n, on=key, how="left")

    a = pd.DataFrame(ablation_rows)
    if not a.empty:
        a = a.loc[a["comparison"] == "ridge_fusion_minus_ridge_ar"].copy()
        a["delta_rmse"] = pd.to_numeric(a["delta_rmse"], errors="coerce")
        win = a.assign(win=(a["delta_rmse"] < 0).astype(float)).groupby(key)["win"].mean().reset_index()
        win = win.rename(columns={"win": "frac_fusion_better_than_ar"})
        out = out.merge(win, on=key, how="left")
    else:
        out["frac_fusion_better_than_ar"] = np.nan

    cols = [
        "subset",
        "context",
        "horizon_weeks",
        "pathogen",
        "n_series",
        "mean_rmse_naive_persistence",
        "mean_rmse_ridge_ar",
        "mean_rmse_ridge_fusion",
        "mean_rmse_ridge_ww_only",
        "frac_fusion_better_than_ar",
    ]
    out = out[cols].sort_values(["context", "horizon_weeks", "pathogen"]).reset_index(drop=True)
    return out.to_dict(orient="records")


if __name__ == "__main__":
    raise SystemExit(main())
