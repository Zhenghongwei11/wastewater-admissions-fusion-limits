#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sign(x: float, *, eps: float) -> int:
    if not np.isfinite(x):
        return 0
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _directional_accuracy(y_true: np.ndarray, y_hat: np.ndarray, *, eps: float) -> float:
    if len(y_true) < 2 or len(y_hat) < 2:
        return float("nan")
    dy = np.diff(y_true)
    dh = np.diff(y_hat)
    s_y = np.asarray([_sign(float(v), eps=eps) for v in dy], dtype=int)
    s_h = np.asarray([_sign(float(v), eps=eps) for v in dh], dtype=int)
    keep = (s_y != 0) & (s_h != 0)
    if int(keep.sum()) == 0:
        return float("nan")
    return float((s_y[keep] == s_h[keep]).mean())


def _peak_week_index(arr: np.ndarray) -> int | None:
    if len(arr) == 0:
        return None
    if not np.isfinite(arr).any():
        return None
    return int(np.nanargmax(arr))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute turning-point style evaluation metrics from long-format forecast predictions (direction-of-change accuracy and peak-timing error)."
    )
    ap.add_argument(
        "--predictions",
        default="results/predictions/state_hosp_predictions.delay0_miss000_rev000.per100k.h1.tsv",
        help="Long-format predictions TSV (default: results/predictions/state_hosp_predictions.delay0_miss000_rev000.per100k.h1.tsv).",
    )
    ap.add_argument("--subset", default="geo_matched_state", help="Subset filter (default: geo_matched_state).")
    ap.add_argument("--horizon", type=int, default=1, help="Horizon (weeks) to evaluate (default: 1).")
    ap.add_argument("--models", default="ridge_ar,ridge_fusion", help="Comma-separated model_ids to evaluate (default: ridge_ar,ridge_fusion).")
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon for near-zero change sign (default: 1e-6).")
    ap.add_argument(
        "--out-by-series",
        default="results/metrics/state_hosp_turning_point_metrics.by_series.per100k.h1.tsv",
        help="Output TSV (per series metrics).",
    )
    ap.add_argument(
        "--out-summary",
        default="results/metrics/state_hosp_turning_point_metrics.summary.per100k.h1.tsv",
        help="Output TSV (summary across series).",
    )
    args = ap.parse_args()

    pred_path = (REPO_ROOT / str(args.predictions)).resolve()
    if not pred_path.exists():
        raise SystemExit(f"missing predictions: {pred_path}")

    df = pd.read_csv(pred_path, sep="\t", dtype={"subset": str, "context": str, "geo_id": str, "pathogen": str, "model_id": str})
    needed = {"subset", "context", "geo_level", "geo_id", "pathogen", "horizon_weeks", "model_id", "week_end", "y_true", "y_hat"}
    miss = sorted(needed - set(df.columns))
    if miss:
        raise SystemExit(f"predictions missing columns: {', '.join(miss)}")

    df = df[(df["subset"].astype(str) == str(args.subset)) & (df["geo_level"].astype(str) == "state")].copy()
    df["horizon_weeks"] = pd.to_numeric(df["horizon_weeks"], errors="coerce")
    df = df[df["horizon_weeks"] == int(args.horizon)].copy()
    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_hat"] = pd.to_numeric(df["y_hat"], errors="coerce")
    df = df.dropna(subset=["week_end", "y_true", "y_hat"]).copy()

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not models:
        raise SystemExit("--models must be non-empty")
    df = df[df["model_id"].isin(models)].copy()

    rows: list[dict[str, object]] = []
    for (ctx, geo_id, pathogen), g in df.groupby(["context", "geo_id", "pathogen"], sort=True):
        g = g.sort_values(["week_end", "model_id"]).copy()
        # y_true is duplicated across models; take the first per week.
        truth = g.drop_duplicates(subset=["week_end"])[["week_end", "y_true"]].sort_values("week_end")
        y_true = truth["y_true"].to_numpy(dtype=float)
        peak_true = _peak_week_index(y_true)
        for model_id, gm in g.groupby("model_id", sort=True):
            gm = gm.sort_values("week_end")
            y_hat = gm["y_hat"].to_numpy(dtype=float)
            if len(y_hat) != len(y_true):
                # Guard against missing rows; align by week_end.
                merged = truth.merge(gm[["week_end", "y_hat"]], on="week_end", how="inner").sort_values("week_end")
                y_true_m = merged["y_true"].to_numpy(dtype=float)
                y_hat_m = merged["y_hat"].to_numpy(dtype=float)
            else:
                y_true_m = y_true
                y_hat_m = y_hat

            peak_pred = _peak_week_index(y_hat_m)
            peak_err = float(abs(int(peak_pred) - int(peak_true))) if peak_pred is not None and peak_true is not None else float("nan")
            dir_acc = _directional_accuracy(y_true_m, y_hat_m, eps=float(args.eps))
            rows.append(
                {
                    "subset": str(args.subset),
                    "context": str(ctx),
                    "geo_level": "state",
                    "geo_id": str(geo_id),
                    "pathogen": str(pathogen),
                    "horizon_weeks": int(args.horizon),
                    "model_id": str(model_id),
                    "n_weeks": int(len(y_true_m)),
                    "dir_acc_nonzero": float(dir_acc) if np.isfinite(dir_acc) else "",
                    "peak_week_index_true": int(peak_true) if peak_true is not None else "",
                    "peak_week_index_pred": int(peak_pred) if peak_pred is not None else "",
                    "peak_timing_abs_error_weeks": float(peak_err) if np.isfinite(peak_err) else "",
                }
            )

    out_by = (REPO_ROOT / str(args.out_by_series)).resolve()
    out_by.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["context", "pathogen", "geo_id", "model_id"]).to_csv(out_by, sep="\t", index=False)

    d = pd.DataFrame(rows)
    # Summaries: median direction accuracy and median peak timing error, by context/pathogen/model.
    for col in ["dir_acc_nonzero", "peak_timing_abs_error_weeks"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    summary = (
        d.groupby(["context", "pathogen", "model_id"], as_index=False)
        .agg(
            n_series=("geo_id", "size"),
            median_dir_acc_nonzero=("dir_acc_nonzero", "median"),
            mean_dir_acc_nonzero=("dir_acc_nonzero", "mean"),
            median_peak_timing_abs_error_weeks=("peak_timing_abs_error_weeks", "median"),
            mean_peak_timing_abs_error_weeks=("peak_timing_abs_error_weeks", "mean"),
        )
        .sort_values(["context", "pathogen", "model_id"])
        .reset_index(drop=True)
    )

    out_s = (REPO_ROOT / str(args.out_summary)).resolve()
    out_s.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_s, sep="\t", index=False)

    print(f"OK turning_point_metrics: wrote {out_by.relative_to(REPO_ROOT)} and {out_s.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

