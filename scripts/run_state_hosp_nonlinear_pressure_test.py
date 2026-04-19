from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid flooding stdout/stderr during long benchmark runs.
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`",
)


@dataclass(frozen=True)
class Config:
    panel_path: str
    subset: str
    y_col: str
    ww_col: str
    horizon_weeks: int
    y_lags: int
    ww_lags: int
    test_weeks: int
    min_weeks: int
    ridge_alpha: float
    rf_n_estimators: int
    rf_random_state: int
    tune_val_weeks: int
    rf_max_depth_grid: tuple[int, ...]
    rf_min_samples_leaf_grid: tuple[int, ...]
    rf_max_features_grid: tuple[float, ...]
    hgb_max_depth_grid: tuple[int, ...]
    hgb_learning_rate_grid: tuple[float, ...]
    hgb_max_leaf_nodes_grid: tuple[int, ...]
    xgb_n_estimators: int
    xgb_random_state: int
    xgb_early_stopping_rounds: int
    xgb_tune_series: int
    xgb_max_depth_grid: tuple[int, ...]
    xgb_learning_rate_grid: tuple[float, ...]
    xgb_subsample_grid: tuple[float, ...]
    xgb_colsample_bytree_grid: tuple[float, ...]
    xgb_min_child_weight_grid: tuple[float, ...]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _build_features(
    y: np.ndarray,
    ww: np.ndarray,
    *,
    horizon_weeks: int,
    y_lags: int,
    ww_lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_lags = max(int(y_lags), int(ww_lags))
    X_ar: list[list[float]] = []
    X_fusion: list[list[float]] = []
    Y: list[float] = []
    for i in range(max_lags, len(y) - int(horizon_weeks)):
        ar_feat = [float(y[i - j]) for j in range(int(y_lags))]
        ww_feat = [float(ww[i - j]) for j in range(int(ww_lags))]
        X_ar.append(ar_feat)
        X_fusion.append(ar_feat + ww_feat)
        Y.append(float(y[i + int(horizon_weeks)]))
    return np.asarray(X_ar), np.asarray(X_fusion), np.asarray(Y)


def _split_train_val(
    X: np.ndarray, Y: np.ndarray, *, val_weeks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_weeks = int(val_weeks)
    if val_weeks <= 0:
        return X, Y, X[:0], Y[:0]
    if len(Y) <= val_weeks + 5:
        # Not enough points to tune in a meaningful way; skip tuning.
        return X, Y, X[:0], Y[:0]
    cut = len(Y) - val_weeks
    return X[:cut], Y[:cut], X[cut:], Y[cut:]


def _tune_rf(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    n_estimators: int,
    random_state: int,
    max_depth_grid: tuple[int, ...],
    min_samples_leaf_grid: tuple[int, ...],
    max_features_grid: tuple[float, ...],
    val_weeks: int,
) -> tuple[dict, RandomForestRegressor]:
    X_fit, Y_fit, X_val, Y_val = _split_train_val(X_train, Y_train, val_weeks=val_weeks)

    # If we cannot form a validation slice, fall back to a conservative default.
    if len(Y_val) == 0:
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": 8,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        }
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=int(random_state),
            n_jobs=-1,
        )
        model.fit(X_train, Y_train)
        return params, model

    best = (float("inf"), None)
    for max_depth in max_depth_grid:
        for min_samples_leaf in min_samples_leaf_grid:
            for max_features in max_features_grid:
                m = RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=float(max_features),
                    random_state=int(random_state),
                    n_jobs=-1,
                )
                m.fit(X_fit, Y_fit)
                rmse = _rmse(Y_val, m.predict(X_val))
                if rmse < best[0]:
                    best = (
                        rmse,
                        {
                            "n_estimators": int(n_estimators),
                            "max_depth": int(max_depth),
                            "min_samples_leaf": int(min_samples_leaf),
                            "max_features": float(max_features),
                        },
                    )

    assert best[1] is not None
    params = best[1]
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=int(random_state),
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)
    return params, model


def _tune_hgb(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    max_depth_grid: tuple[int, ...],
    learning_rate_grid: tuple[float, ...],
    max_leaf_nodes_grid: tuple[int, ...],
    val_weeks: int,
) -> tuple[dict, HistGradientBoostingRegressor]:
    X_fit, Y_fit, X_val, Y_val = _split_train_val(X_train, Y_train, val_weeks=val_weeks)

    if len(Y_val) == 0:
        params = {"max_depth": 3, "learning_rate": 0.05, "max_leaf_nodes": 31}
        model = HistGradientBoostingRegressor(
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            random_state=0,
        )
        model.fit(X_train, Y_train)
        return params, model

    best = (float("inf"), None)
    for max_depth in max_depth_grid:
        for lr in learning_rate_grid:
            for mln in max_leaf_nodes_grid:
                m = HistGradientBoostingRegressor(
                    max_depth=int(max_depth),
                    learning_rate=float(lr),
                    max_leaf_nodes=int(mln),
                    random_state=0,
                )
                m.fit(X_fit, Y_fit)
                rmse = _rmse(Y_val, m.predict(X_val))
                if rmse < best[0]:
                    best = (rmse, {"max_depth": int(max_depth), "learning_rate": float(lr), "max_leaf_nodes": int(mln)})

    assert best[1] is not None
    params = best[1]
    model = HistGradientBoostingRegressor(
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        random_state=0,
    )
    model.fit(X_train, Y_train)
    return params, model


def _tune_xgb(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    n_estimators: int,
    random_state: int,
    early_stopping_rounds: int,
    max_depth_grid: tuple[int, ...],
    learning_rate_grid: tuple[float, ...],
    subsample_grid: tuple[float, ...],
    colsample_bytree_grid: tuple[float, ...],
    min_child_weight_grid: tuple[float, ...],
    val_weeks: int,
) -> tuple[dict, "XGBRegressor"]:
    if XGBRegressor is None:
        raise RuntimeError(
            "xgboost could not be imported. On macOS this often means the OpenMP runtime is missing (libomp). "
            "Install it (e.g., `brew install libomp`) and retry."
        )

    X_fit, Y_fit, X_val, Y_val = _split_train_val(X_train, Y_train, val_weeks=val_weeks)
    if len(Y_val) == 0:
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 1.0,
            "min_child_weight": 1.0,
        }
        model = XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            objective="reg:squarederror",
            tree_method="hist",
            random_state=int(random_state),
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, Y_train)
        return params, model

    # NOTE: Per-series grid-search is expensive; keep the grid intentionally small.
    best = (float("inf"), None, None, None)  # (rmse, params, best_iteration, fitted_model)
    for max_depth in max_depth_grid:
        for lr in learning_rate_grid:
            for subsample in subsample_grid:
                for colsample in colsample_bytree_grid:
                    for mcw in min_child_weight_grid:
                        m = XGBRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=int(max_depth),
                            learning_rate=float(lr),
                            subsample=float(subsample),
                            colsample_bytree=float(colsample),
                            min_child_weight=float(mcw),
                            early_stopping_rounds=int(early_stopping_rounds),
                            objective="reg:squarederror",
                            tree_method="hist",
                            random_state=int(random_state),
                            n_jobs=-1,
                            verbosity=0,
                        )
                        m.fit(X_fit, Y_fit, eval_set=[(X_val, Y_val)], verbose=False)
                        rmse = _rmse(Y_val, m.predict(X_val))
                        best_iter = getattr(m, "best_iteration", None)
                        if rmse < best[0]:
                            best = (
                                rmse,
                                {
                                    "n_estimators": int(n_estimators),
                                    "max_depth": int(max_depth),
                                    "learning_rate": float(lr),
                                    "subsample": float(subsample),
                                    "colsample_bytree": float(colsample),
                                    "min_child_weight": float(mcw),
                                },
                                (int(best_iter) if best_iter is not None else None),
                                m,
                            )

    assert best[1] is not None and best[3] is not None
    params = best[1]
    best_iter = best[2]
    model = best[3]

    # For speed, keep the early-stopped model trained on the time-ordered fit period.
    # We record the selected effective number of boosting rounds for transparency.
    final_n_estimators = int(best_iter + 1) if (best_iter is not None and best_iter >= 0) else int(params["n_estimators"])
    params_out = dict(params)
    params_out["n_estimators"] = int(final_n_estimators)
    if best_iter is not None:
        params_out["early_stopping_best_iteration"] = int(best_iter)
    params_out["trained_on_fit_window_only"] = True
    return params_out, model


@dataclass(frozen=True)
class SeriesItem:
    geo_id: str
    pathogen: str
    X_ar_train: np.ndarray
    X_ar_test: np.ndarray
    X_fusion_train: np.ndarray
    X_fusion_test: np.ndarray
    Y_train: np.ndarray
    Y_test: np.ndarray


def _xgb_candidate_grid(cfg: Config) -> list[dict]:
    cands: list[dict] = []
    for max_depth in cfg.xgb_max_depth_grid:
        for lr in cfg.xgb_learning_rate_grid:
            for subsample in cfg.xgb_subsample_grid:
                for colsample in cfg.xgb_colsample_bytree_grid:
                    for mcw in cfg.xgb_min_child_weight_grid:
                        cands.append(
                            {
                                "max_depth": int(max_depth),
                                "learning_rate": float(lr),
                                "subsample": float(subsample),
                                "colsample_bytree": float(colsample),
                                "min_child_weight": float(mcw),
                            }
                        )
    return cands


def run(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(cfg.panel_path, sep="\t")
    df["week_end"] = pd.to_datetime(df["week_end"])
    df = df.dropna(subset=[cfg.y_col, cfg.ww_col]).copy()
    df["ww_log"] = np.log10(1.0 + np.clip(pd.to_numeric(df[cfg.ww_col], errors="coerce"), 0, None))
    df = df.dropna(subset=["ww_log"])

    items: list[SeriesItem] = []
    for (geo_id, pathogen), g in df.groupby(["geo_id", "pathogen"]):
        g = g.sort_values("week_end").reset_index(drop=True)
        if len(g) < int(cfg.min_weeks):
            continue

        y = pd.to_numeric(g[cfg.y_col], errors="coerce").to_numpy(dtype=float)
        ww = pd.to_numeric(g["ww_log"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any() or np.isnan(ww).any():
            continue

        X_ar, X_fusion, Y = _build_features(y, ww, horizon_weeks=cfg.horizon_weeks, y_lags=cfg.y_lags, ww_lags=cfg.ww_lags)
        if len(Y) <= int(cfg.test_weeks) + 1:
            continue

        split = len(Y) - int(cfg.test_weeks)
        items.append(
            SeriesItem(
                geo_id=str(geo_id),
                pathogen=str(pathogen),
                X_ar_train=X_ar[:split],
                X_ar_test=X_ar[split:],
                X_fusion_train=X_fusion[:split],
                X_fusion_test=X_fusion[split:],
                Y_train=Y[:split],
                Y_test=Y[split:],
            )
        )

    items = sorted(items, key=lambda it: (it.geo_id, it.pathogen))

    # Global XGBoost tuning on a small deterministic subset (addresses "not tuned" critique
    # without making the benchmark itself an HPO study).
    xgb_global = {"enabled": False}
    xgb_ar_base = None
    xgb_fusion_base = None
    if XGBRegressor is not None and len(items) > 0 and int(cfg.xgb_tune_series) > 0:
        tune_items = items[: min(int(cfg.xgb_tune_series), len(items))]
        cands = _xgb_candidate_grid(cfg)
        if len(cands) == 0:
            raise RuntimeError("Empty XGBoost candidate grid.")

        def score_candidate(candidate: dict, *, use_fusion: bool) -> float:
            rmses = []
            for it in tune_items:
                X_train = it.X_fusion_train if use_fusion else it.X_ar_train
                Y_train = it.Y_train
                X_fit, Y_fit, X_val, Y_val = _split_train_val(X_train, Y_train, val_weeks=int(cfg.tune_val_weeks))
                if len(Y_val) == 0:
                    continue
                m = XGBRegressor(
                    n_estimators=int(cfg.xgb_n_estimators),
                    max_depth=int(candidate["max_depth"]),
                    learning_rate=float(candidate["learning_rate"]),
                    subsample=float(candidate["subsample"]),
                    colsample_bytree=float(candidate["colsample_bytree"]),
                    min_child_weight=float(candidate["min_child_weight"]),
                    early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=int(cfg.xgb_random_state),
                    n_jobs=-1,
                    verbosity=0,
                )
                m.fit(X_fit, Y_fit, eval_set=[(X_val, Y_val)], verbose=False)
                rmses.append(_rmse(Y_val, m.predict(X_val)))
            return float(np.mean(rmses)) if rmses else float("inf")

        scored_ar = [(score_candidate(c, use_fusion=False), c) for c in cands]
        scored_fusion = [(score_candidate(c, use_fusion=True), c) for c in cands]
        scored_ar.sort(key=lambda t: t[0])
        scored_fusion.sort(key=lambda t: t[0])
        xgb_ar_base = scored_ar[0][1]
        xgb_fusion_base = scored_fusion[0][1]
        xgb_global = {
            "enabled": True,
            "tune_series": int(len(tune_items)),
            "n_candidates": int(len(cands)),
            "best_ar_candidate": dict(xgb_ar_base),
            "best_ar_val_rmse_mean": float(scored_ar[0][0]),
            "best_fusion_candidate": dict(xgb_fusion_base),
            "best_fusion_val_rmse_mean": float(scored_fusion[0][0]),
        }

    rows: list[dict] = []
    for it in items:
        X_ar_train, X_ar_test = it.X_ar_train, it.X_ar_test
        X_fs_train, X_fs_test = it.X_fusion_train, it.X_fusion_test
        Y_train, Y_test = it.Y_train, it.Y_test

        ridge_ar = Ridge(alpha=float(cfg.ridge_alpha))
        ridge_fusion = Ridge(alpha=float(cfg.ridge_alpha))

        ridge_ar.fit(X_ar_train, Y_train)
        ridge_fusion.fit(X_fs_train, Y_train)

        rmse_ridge_ar = _rmse(Y_test, ridge_ar.predict(X_ar_test))
        rmse_ridge_fusion = _rmse(Y_test, ridge_fusion.predict(X_fs_test))

        rf_ar_params, rf_ar = _tune_rf(
            X_ar_train,
            Y_train,
            n_estimators=int(cfg.rf_n_estimators),
            random_state=int(cfg.rf_random_state),
            max_depth_grid=cfg.rf_max_depth_grid,
            min_samples_leaf_grid=cfg.rf_min_samples_leaf_grid,
            max_features_grid=cfg.rf_max_features_grid,
            val_weeks=int(cfg.tune_val_weeks),
        )
        rf_fusion_params, rf_fusion = _tune_rf(
            X_fs_train,
            Y_train,
            n_estimators=int(cfg.rf_n_estimators),
            random_state=int(cfg.rf_random_state),
            max_depth_grid=cfg.rf_max_depth_grid,
            min_samples_leaf_grid=cfg.rf_min_samples_leaf_grid,
            max_features_grid=cfg.rf_max_features_grid,
            val_weeks=int(cfg.tune_val_weeks),
        )

        rmse_rf_ar = _rmse(Y_test, rf_ar.predict(X_ar_test))
        rmse_rf_fusion = _rmse(Y_test, rf_fusion.predict(X_fs_test))

        hgb_ar_params, hgb_ar = _tune_hgb(
            X_ar_train,
            Y_train,
            max_depth_grid=cfg.hgb_max_depth_grid,
            learning_rate_grid=cfg.hgb_learning_rate_grid,
            max_leaf_nodes_grid=cfg.hgb_max_leaf_nodes_grid,
            val_weeks=int(cfg.tune_val_weeks),
        )
        hgb_fusion_params, hgb_fusion = _tune_hgb(
            X_fs_train,
            Y_train,
            max_depth_grid=cfg.hgb_max_depth_grid,
            learning_rate_grid=cfg.hgb_learning_rate_grid,
            max_leaf_nodes_grid=cfg.hgb_max_leaf_nodes_grid,
            val_weeks=int(cfg.tune_val_weeks),
        )
        rmse_hgb_ar = _rmse(Y_test, hgb_ar.predict(X_ar_test))
        rmse_hgb_fusion = _rmse(Y_test, hgb_fusion.predict(X_fs_test))

        xgb_ar_params, xgb_ar = _tune_xgb(
            X_ar_train,
            Y_train,
            n_estimators=int(cfg.xgb_n_estimators),
            random_state=int(cfg.xgb_random_state),
            early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
            max_depth_grid=cfg.xgb_max_depth_grid,
            learning_rate_grid=cfg.xgb_learning_rate_grid,
            subsample_grid=cfg.xgb_subsample_grid,
            colsample_bytree_grid=cfg.xgb_colsample_bytree_grid,
            min_child_weight_grid=cfg.xgb_min_child_weight_grid,
            val_weeks=int(cfg.tune_val_weeks),
        )
        xgb_fusion_params, xgb_fusion = _tune_xgb(
            X_fs_train,
            Y_train,
            n_estimators=int(cfg.xgb_n_estimators),
            random_state=int(cfg.xgb_random_state),
            early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
            max_depth_grid=cfg.xgb_max_depth_grid,
            learning_rate_grid=cfg.xgb_learning_rate_grid,
            subsample_grid=cfg.xgb_subsample_grid,
            colsample_bytree_grid=cfg.xgb_colsample_bytree_grid,
            min_child_weight_grid=cfg.xgb_min_child_weight_grid,
            val_weeks=int(cfg.tune_val_weeks),
        )
        rmse_xgb_ar = _rmse(Y_test, xgb_ar.predict(X_ar_test))
        rmse_xgb_fusion = _rmse(Y_test, xgb_fusion.predict(X_fs_test))

        rmse_xgb_ar = float("nan")
        rmse_xgb_fusion = float("nan")
        xgb_ar_params_json = ""
        xgb_fusion_params_json = ""
        xgb_fusion_better = 0
        if xgb_global.get("enabled") and xgb_ar_base is not None and xgb_fusion_base is not None:
            xgb_ar_params, xgb_ar = _tune_xgb(
                X_ar_train,
                Y_train,
                n_estimators=int(cfg.xgb_n_estimators),
                random_state=int(cfg.xgb_random_state),
                early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
                max_depth_grid=(int(xgb_ar_base["max_depth"]),),
                learning_rate_grid=(float(xgb_ar_base["learning_rate"]),),
                subsample_grid=(float(xgb_ar_base["subsample"]),),
                colsample_bytree_grid=(float(xgb_ar_base["colsample_bytree"]),),
                min_child_weight_grid=(float(xgb_ar_base["min_child_weight"]),),
                val_weeks=int(cfg.tune_val_weeks),
            )
            xgb_fusion_params, xgb_fusion = _tune_xgb(
                X_fs_train,
                Y_train,
                n_estimators=int(cfg.xgb_n_estimators),
                random_state=int(cfg.xgb_random_state),
                early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
                max_depth_grid=(int(xgb_fusion_base["max_depth"]),),
                learning_rate_grid=(float(xgb_fusion_base["learning_rate"]),),
                subsample_grid=(float(xgb_fusion_base["subsample"]),),
                colsample_bytree_grid=(float(xgb_fusion_base["colsample_bytree"]),),
                min_child_weight_grid=(float(xgb_fusion_base["min_child_weight"]),),
                val_weeks=int(cfg.tune_val_weeks),
            )
            rmse_xgb_ar = _rmse(Y_test, xgb_ar.predict(X_ar_test))
            rmse_xgb_fusion = _rmse(Y_test, xgb_fusion.predict(X_fs_test))
            xgb_ar_params_json = json.dumps(xgb_ar_params, sort_keys=True)
            xgb_fusion_params_json = json.dumps(xgb_fusion_params, sort_keys=True)
            xgb_fusion_better = int(rmse_xgb_fusion < rmse_xgb_ar)

        rows.append(
            {
                "subset": cfg.subset,
                "geo_id": it.geo_id,
                "pathogen": it.pathogen,
                "horizon_weeks": int(cfg.horizon_weeks),
                "test_weeks": int(cfg.test_weeks),
                "rmse_ridge_ar": rmse_ridge_ar,
                "rmse_ridge_fusion": rmse_ridge_fusion,
                "rmse_rf_tuned_ar": rmse_rf_ar,
                "rmse_rf_tuned_fusion": rmse_rf_fusion,
                "rf_tuned_ar_params_json": json.dumps(rf_ar_params, sort_keys=True),
                "rf_tuned_fusion_params_json": json.dumps(rf_fusion_params, sort_keys=True),
                "rmse_hgb_ar": rmse_hgb_ar,
                "rmse_hgb_fusion": rmse_hgb_fusion,
                "hgb_ar_params_json": json.dumps(hgb_ar_params, sort_keys=True),
                "hgb_fusion_params_json": json.dumps(hgb_fusion_params, sort_keys=True),
                "rmse_xgb_ar": rmse_xgb_ar,
                "rmse_xgb_fusion": rmse_xgb_fusion,
                "xgb_ar_params_json": xgb_ar_params_json,
                "xgb_fusion_params_json": xgb_fusion_params_json,
                "ridge_fusion_better_than_ar": int(rmse_ridge_fusion < rmse_ridge_ar),
                "rf_tuned_fusion_better_than_ar": int(rmse_rf_fusion < rmse_rf_ar),
                "hgb_fusion_better_than_ar": int(rmse_hgb_fusion < rmse_hgb_ar),
                "xgb_fusion_better_than_ar": xgb_fusion_better,
            }
        )

    by_series = pd.DataFrame(rows).sort_values(["subset", "geo_id", "pathogen"]).reset_index(drop=True)
    summary_row: dict[str, float] = {
        "subset": cfg.subset,
        "n_series": float(len(by_series)),
        "mean_rmse_ridge_ar": float(by_series["rmse_ridge_ar"].mean()) if len(by_series) else float("nan"),
        "mean_rmse_ridge_fusion": float(by_series["rmse_ridge_fusion"].mean()) if len(by_series) else float("nan"),
        "mean_rmse_rf_tuned_ar": float(by_series["rmse_rf_tuned_ar"].mean()) if len(by_series) else float("nan"),
        "mean_rmse_rf_tuned_fusion": float(by_series["rmse_rf_tuned_fusion"].mean()) if len(by_series) else float("nan"),
        "mean_rmse_hgb_ar": float(by_series["rmse_hgb_ar"].mean()) if len(by_series) else float("nan"),
        "mean_rmse_hgb_fusion": float(by_series["rmse_hgb_fusion"].mean()) if len(by_series) else float("nan"),
        "mean_rmse_xgb_ar": float(by_series["rmse_xgb_ar"].mean()) if len(by_series) else float("nan"),
        "mean_rmse_xgb_fusion": float(by_series["rmse_xgb_fusion"].mean()) if len(by_series) else float("nan"),
        "ridge_fusion_win_rate": float(by_series["ridge_fusion_better_than_ar"].mean()) if len(by_series) else float("nan"),
        "rf_tuned_fusion_win_rate": float(by_series["rf_tuned_fusion_better_than_ar"].mean()) if len(by_series) else float("nan"),
        "hgb_fusion_win_rate": float(by_series["hgb_fusion_better_than_ar"].mean()) if len(by_series) else float("nan"),
        "xgb_fusion_win_rate": float(by_series["xgb_fusion_better_than_ar"].mean()) if len(by_series) else float("nan"),
    }
    summary = pd.DataFrame([summary_row])

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "xgb_global_tuning": xgb_global,
    }
    return by_series, summary, meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Route A model-family comparison: Ridge vs tuned tree-based models (RF + gradient boosting), leakage-safe temporal split."
    )
    ap.add_argument("--panel", default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv", help="Input panel TSV.")
    ap.add_argument("--subset", default="geo_matched_state", help="Subset label to write into outputs.")
    ap.add_argument("--y-col", default="hosp_admissions_per_100k", help="Admissions target column.")
    ap.add_argument("--ww-col", default="nwss_conc_mean", help="Wastewater concentration column.")
    ap.add_argument("--horizon-weeks", type=int, default=1, help="Forecast horizon in weeks (default: 1).")
    ap.add_argument("--y-lags", type=int, default=4, help="Number of AR lags (default: 4).")
    ap.add_argument("--ww-lags", type=int, default=4, help="Number of wastewater lags (default: 4).")
    ap.add_argument("--test-weeks", type=int, default=52, help="Holdout weeks per series (default: 52).")
    ap.add_argument("--min-weeks", type=int, default=60, help="Minimum raw weeks required per series before feature construction (default: 60).")
    ap.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0).")
    ap.add_argument("--rf-n-estimators", type=int, default=150, help="RF n_estimators (default: 150).")
    ap.add_argument("--rf-random-state", type=int, default=42, help="RF random_state (default: 42).")
    ap.add_argument("--tune-val-weeks", type=int, default=26, help="Tuning validation weeks taken from end of training (default: 26).")
    ap.add_argument("--xgb-n-estimators", type=int, default=600, help="XGBoost n_estimators (upper bound; default: 600).")
    ap.add_argument("--xgb-random-state", type=int, default=42, help="XGBoost random_state (default: 42).")
    ap.add_argument("--xgb-early-stopping-rounds", type=int, default=25, help="XGBoost early stopping rounds (default: 25).")
    ap.add_argument("--xgb-tune-series", type=int, default=12, help="Number of series used for global XGBoost tuning (default: 12).")
    ap.add_argument("--out-by-series", default="results/benchmark/state_hosp_nonlinear_pressure_test.by_series.tsv", help="By-series TSV output.")
    ap.add_argument("--out-summary", default="results/benchmark/state_hosp_nonlinear_pressure_test.summary.tsv", help="Summary TSV output.")
    ap.add_argument("--out-meta", default=None, help="Optional meta JSON output path (default: <out-summary>.meta.json).")
    args = ap.parse_args()

    cfg = Config(
        panel_path=str(args.panel),
        subset=str(args.subset),
        y_col=str(args.y_col),
        ww_col=str(args.ww_col),
        horizon_weeks=int(args.horizon_weeks),
        y_lags=int(args.y_lags),
        ww_lags=int(args.ww_lags),
        test_weeks=int(args.test_weeks),
        min_weeks=int(args.min_weeks),
        ridge_alpha=float(args.ridge_alpha),
        rf_n_estimators=int(args.rf_n_estimators),
        rf_random_state=int(args.rf_random_state),
        tune_val_weeks=int(args.tune_val_weeks),
        # Keep tuning lightweight: a small, time-ordered grid to address reviewer concerns
        # about under-tuned non-linear baselines without turning the analysis into
        # a hyperparameter-optimization study.
        rf_max_depth_grid=(4, 8),
        rf_min_samples_leaf_grid=(1, 5),
        rf_max_features_grid=(1.0,),
        hgb_max_depth_grid=(3, 5),
        hgb_learning_rate_grid=(0.05, 0.10),
        hgb_max_leaf_nodes_grid=(31,),
        xgb_n_estimators=int(args.xgb_n_estimators),
        xgb_random_state=int(args.xgb_random_state),
        xgb_early_stopping_rounds=int(args.xgb_early_stopping_rounds),
        xgb_tune_series=int(args.xgb_tune_series),
        xgb_max_depth_grid=(3, 5),
        xgb_learning_rate_grid=(0.05, 0.10),
        xgb_subsample_grid=(1.0,),
        xgb_colsample_bytree_grid=(1.0,),
        xgb_min_child_weight_grid=(1.0,),
    )

    by_series, summary, meta = run(cfg)

    out_by = (REPO_ROOT / str(args.out_by_series)).resolve()
    out_by.parent.mkdir(parents=True, exist_ok=True)
    by_series.to_csv(out_by, sep="\t", index=False)

    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, sep="\t", index=False)

    out_meta = (REPO_ROOT / (str(args.out_meta) if args.out_meta else (str(out_summary) + ".meta.json"))).resolve()
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Print reviewer-friendly summary (matches the legacy quick script numbers).
    s = summary.iloc[0].to_dict()
    print("\n--- Nonlinear Verification Summary ---")
    print(f"Mean_Ridge_AR: {float(s['mean_rmse_ridge_ar']):.4f}")
    print(f"Mean_Ridge_Fusion: {float(s['mean_rmse_ridge_fusion']):.4f}")
    print(f"Mean_RF_Tuned_AR: {float(s['mean_rmse_rf_tuned_ar']):.4f}")
    print(f"Mean_RF_Tuned_Fusion: {float(s['mean_rmse_rf_tuned_fusion']):.4f}")
    print(f"Mean_HGB_AR: {float(s['mean_rmse_hgb_ar']):.4f}")
    print(f"Mean_HGB_Fusion: {float(s['mean_rmse_hgb_fusion']):.4f}")
    print(f"Mean_XGB_AR: {float(s['mean_rmse_xgb_ar']):.4f}")
    print(f"Mean_XGB_Fusion: {float(s['mean_rmse_xgb_fusion']):.4f}")
    print(f"Ridge_Fusion_Win_Rate: {float(s['ridge_fusion_win_rate']):.4f}")
    print(f"RF_Tuned_Fusion_Win_Rate: {float(s['rf_tuned_fusion_win_rate']):.4f}")
    print(f"HGB_Fusion_Win_Rate: {float(s['hgb_fusion_win_rate']):.4f}")
    print(f"XGB_Fusion_Win_Rate: {float(s['xgb_fusion_win_rate']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
