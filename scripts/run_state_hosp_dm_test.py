#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    loss: str  # "se" or "ae"


def _build_features(
    y: np.ndarray,
    ww: np.ndarray,
    *,
    horizon_weeks: int,
    y_lags: int,
    ww_lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_lags = max(int(y_lags), int(ww_lags))
    X_ar: list[list[float]] = []
    X_fusion: list[list[float]] = []
    Y: list[float] = []
    week_idx: list[int] = []
    for i in range(max_lags, len(y) - int(horizon_weeks)):
        ar_feat = [float(y[i - j]) for j in range(int(y_lags))]
        ww_feat = [float(ww[i - j]) for j in range(int(ww_lags))]
        X_ar.append(ar_feat)
        X_fusion.append(ar_feat + ww_feat)
        Y.append(float(y[i + int(horizon_weeks)]))
        week_idx.append(i + int(horizon_weeks))
    return np.asarray(X_ar), np.asarray(X_fusion), np.asarray(Y), np.asarray(week_idx, dtype=int)


def _dm_test(d: np.ndarray, *, h: int) -> tuple[float, float]:
    """
    Diebold–Mariano test (two-sided) using a Newey–West variance estimate.

    Inputs:
      d_t = loss_fusion(t) - loss_baseline(t). Positive mean => baseline better.

    For h-step-ahead forecasts, a common choice is lag = h-1. Here h>=1.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.shape[0])
    if n < 8:
        return float("nan"), float("nan")

    mean_d = float(d.mean())
    lag = max(int(h) - 1, 0)

    # Newey–West (Bartlett) estimate of var(mean_d).
    gamma0 = float(np.mean((d - mean_d) ** 2))
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - (k / (lag + 1.0))
        cov = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        var += 2.0 * w * cov
    var_mean = var / n
    if not (var_mean > 0):
        return mean_d, float("nan")

    dm_stat = mean_d / math.sqrt(var_mean)

    # Normal approximation for p-value (sufficient here; we report it descriptively).
    # p = 2 * (1 - Phi(|z|))
    z = abs(dm_stat)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return mean_d, float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute per-series Diebold–Mariano tests: Ridge AR vs Ridge fusion (state panel).")
    ap.add_argument("--panel", default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv", help="Input panel TSV.")
    ap.add_argument("--subset", default="geo_matched_state", help="Subset label written into outputs.")
    ap.add_argument("--y-col", default="hosp_admissions_per_100k", help="Admissions target column.")
    ap.add_argument("--ww-col", default="nwss_conc_mean", help="Wastewater concentration column.")
    ap.add_argument("--horizon-weeks", type=int, default=1, help="Forecast horizon in weeks (default: 1).")
    ap.add_argument("--y-lags", type=int, default=4, help="Number of AR lags (default: 4).")
    ap.add_argument("--ww-lags", type=int, default=4, help="Number of wastewater lags (default: 4).")
    ap.add_argument("--test-weeks", type=int, default=52, help="Holdout weeks per series (default: 52).")
    ap.add_argument("--min-weeks", type=int, default=60, help="Minimum raw weeks per series (default: 60).")
    ap.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0).")
    ap.add_argument("--loss", default="se", choices=["se", "ae"], help="Loss for DM test: squared error (se) or absolute error (ae).")
    ap.add_argument("--out", default="results/metrics/state_hosp_dm_test.per100k.tsv", help="Per-series DM output TSV.")
    ap.add_argument("--out-summary", default="results/metrics/state_hosp_dm_test_summary.per100k.tsv", help="Summary TSV output.")
    ap.add_argument("--out-meta", default=None, help="Optional meta JSON output path.")
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
        loss=str(args.loss),
    )

    panel_path = (REPO_ROOT / cfg.panel_path).resolve()
    df = pd.read_csv(panel_path, sep="\t")
    df["week_end"] = pd.to_datetime(df["week_end"])
    df = df.dropna(subset=[cfg.y_col, cfg.ww_col]).copy()
    df["ww_log"] = np.log10(1.0 + np.clip(pd.to_numeric(df[cfg.ww_col], errors="coerce"), 0, None))
    df = df.dropna(subset=["ww_log"])

    rows: list[dict] = []
    for (geo_id, pathogen), g in df.groupby(["geo_id", "pathogen"]):
        g = g.sort_values("week_end").reset_index(drop=True)
        if len(g) < int(cfg.min_weeks):
            continue

        y = pd.to_numeric(g[cfg.y_col], errors="coerce").to_numpy(dtype=float)
        ww = pd.to_numeric(g["ww_log"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any() or np.isnan(ww).any():
            continue

        X_ar, X_fusion, Y, week_idx = _build_features(y, ww, horizon_weeks=cfg.horizon_weeks, y_lags=cfg.y_lags, ww_lags=cfg.ww_lags)
        if len(Y) <= int(cfg.test_weeks) + 1:
            continue

        split = len(Y) - int(cfg.test_weeks)
        X_ar_train, X_ar_test = X_ar[:split], X_ar[split:]
        X_fs_train, X_fs_test = X_fusion[:split], X_fusion[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        widx_test = week_idx[split:]

        ridge_ar = Ridge(alpha=float(cfg.ridge_alpha))
        ridge_fusion = Ridge(alpha=float(cfg.ridge_alpha))
        ridge_ar.fit(X_ar_train, Y_train)
        ridge_fusion.fit(X_fs_train, Y_train)

        pred_ar = ridge_ar.predict(X_ar_test)
        pred_fusion = ridge_fusion.predict(X_fs_test)

        if cfg.loss == "se":
            loss_ar = (Y_test - pred_ar) ** 2
            loss_fusion = (Y_test - pred_fusion) ** 2
        else:
            loss_ar = np.abs(Y_test - pred_ar)
            loss_fusion = np.abs(Y_test - pred_fusion)

        d = loss_fusion - loss_ar
        mean_d, p = _dm_test(d, h=int(cfg.horizon_weeks))

        rows.append(
            {
                "subset": cfg.subset,
                "geo_level": "state",
                "geo_id": str(geo_id),
                "pathogen": str(pathogen),
                "horizon_weeks": int(cfg.horizon_weeks),
                "test_weeks": int(cfg.test_weeks),
                "loss": str(cfg.loss),
                "mean_loss_diff_fusion_minus_ar": float(mean_d),
                "dm_p_value_two_sided": float(p),
                "test_start_week_end": str(pd.to_datetime(g["week_end"].iloc[int(widx_test.min())]).date()),
                "test_end_week_end": str(pd.to_datetime(g["week_end"].iloc[int(widx_test.max())]).date()),
            }
        )

    out = (REPO_ROOT / str(args.out)).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    dmf = pd.DataFrame(rows).sort_values(["geo_id", "pathogen"]).reset_index(drop=True)
    dmf.to_csv(out, sep="\t", index=False)

    # Summary: proportion significant in each direction + median effect.
    sig = dmf["dm_p_value_two_sided"] < 0.05
    worse = dmf["mean_loss_diff_fusion_minus_ar"] > 0
    better = dmf["mean_loss_diff_fusion_minus_ar"] < 0
    summary = pd.DataFrame(
        [
            {
                "subset": cfg.subset,
                "loss": cfg.loss,
                "horizon_weeks": int(cfg.horizon_weeks),
                "n_series": int(dmf.shape[0]),
                "median_mean_loss_diff_fusion_minus_ar": float(pd.to_numeric(dmf["mean_loss_diff_fusion_minus_ar"], errors="coerce").median()),
                "frac_series_sig": float(sig.mean()) if len(dmf) else float("nan"),
                "frac_series_sig_fusion_worse": float((sig & worse).mean()) if len(dmf) else float("nan"),
                "frac_series_sig_fusion_better": float((sig & better).mean()) if len(dmf) else float("nan"),
            }
        ]
    )
    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, sep="\t", index=False)

    meta = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg)}
    out_meta = (REPO_ROOT / (str(args.out_meta) if args.out_meta else (str(out_summary) + ".meta.json"))).resolve()
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK DM: wrote {out}")
    print(f"OK DM: wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
