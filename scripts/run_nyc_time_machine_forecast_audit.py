from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Cfg:
    y_lags: int
    ridge_alpha: float
    train_window_weeks: int
    tol_abs: float
    tol_rel: float


def main() -> int:
    ap = argparse.ArgumentParser(description="NYC git time-machine forecast audit (revision-driven prediction stability).")
    ap.add_argument(
        "--trajectory",
        default="results/revision/nyc_ed_respiratory_illness_git_trajectory.tsv",
        help="Long git trajectory table: commit × week_end × metric (default: results/revision/nyc_ed_respiratory_illness_git_trajectory.tsv).",
    )
    ap.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names to include (default: all metrics in the trajectory).",
    )
    ap.add_argument(
        "--origins-since",
        default="2026-01-01",
        help="Only evaluate forecast origins with week_end >= this date (default: 2026-01-01).",
    )
    ap.add_argument(
        "--horizons",
        default="1,2,3,4",
        help="Comma-separated forecast horizons in weeks (default: 1,2,3,4).",
    )
    ap.add_argument("--y-lags", type=int, default=2, help="Autoregressive lags for ridge AR (default: 2).")
    ap.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0).")
    ap.add_argument(
        "--train-window-weeks",
        type=int,
        default=12,
        help="Training window length in weeks prior to each origin (default: 12).",
    )
    ap.add_argument("--tol-abs", type=float, default=0.05, help="Absolute tolerance for forecast stability (default: 0.05).")
    ap.add_argument("--tol-rel", type=float, default=0.01, help="Relative tolerance for forecast stability (default: 0.01).")
    ap.add_argument(
        "--out-preds",
        default="results/revision/nyc_time_machine_forecast_predictions.tsv",
        help="Long predictions output (default: results/revision/nyc_time_machine_forecast_predictions.tsv).",
    )
    ap.add_argument(
        "--out-stability",
        default="results/revision/nyc_time_machine_forecast_stability.tsv",
        help="Stability summary output (default: results/revision/nyc_time_machine_forecast_stability.tsv).",
    )
    ap.add_argument("--out-meta", default=None, help="Optional JSON metadata output path (default: <out-stability>.meta.json).")
    args = ap.parse_args()

    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    horizons = sorted({h for h in horizons if h >= 1})
    if not horizons:
        raise SystemExit("--horizons must include at least one integer >= 1")

    traj_path = (REPO_ROOT / str(args.trajectory)).resolve()
    if not traj_path.exists():
        raise SystemExit(f"missing trajectory table: {traj_path}")

    traj = pd.read_csv(traj_path, sep="\t", dtype={"commit_sha": str, "commit_time_iso": str, "week_end": str, "metric": str})
    req = {"commit_sha", "commit_time_iso", "week_end", "metric", "value"}
    miss = sorted(req - set(traj.columns))
    if miss:
        raise SystemExit(f"trajectory missing columns: {', '.join(miss)}")

    traj["commit_time_utc"] = pd.to_datetime(traj["commit_time_iso"], errors="coerce", utc=True)
    traj["week_end"] = pd.to_datetime(traj["week_end"], errors="coerce")
    traj["value"] = pd.to_numeric(traj["value"], errors="coerce")
    traj = traj.dropna(subset=["commit_time_utc", "week_end", "metric"]).copy()

    metrics = sorted(traj["metric"].astype(str).unique().tolist())
    if args.metrics:
        want = {m.strip() for m in str(args.metrics).split(",") if m.strip()}
        metrics = [m for m in metrics if m in want]
    if not metrics:
        raise SystemExit("no metrics selected")

    origins_since = pd.Timestamp(str(args.origins_since))
    # “final” snapshot = latest commit in the trajectory window.
    final_commit_time = traj["commit_time_utc"].max()
    final_sha = str(traj.loc[traj["commit_time_utc"] == final_commit_time, "commit_sha"].iloc[-1])

    cfg = Cfg(
        y_lags=int(args.y_lags),
        ridge_alpha=float(args.ridge_alpha),
        train_window_weeks=int(args.train_window_weeks),
        tol_abs=float(args.tol_abs),
        tol_rel=float(args.tol_rel),
    )

    out_pred_rows: list[dict[str, Any]] = []

    commits = (
        traj[["commit_sha", "commit_time_utc"]]
        .drop_duplicates()
        .sort_values(["commit_time_utc", "commit_sha"])
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    for metric in metrics:
        # Precompute final snapshot series for truth + final predictions anchor.
        final_series = _series_for_commit(traj, sha=final_sha, metric=metric)
        if final_series.empty:
            continue

        week_ends = final_series["week_end"].to_list()
        origins = [we for we in week_ends if we >= origins_since]
        if not origins:
            continue

        for origin in origins:
            for h in horizons:
                target_week_end = origin + timedelta(days=7 * int(h))
                y_true_final = final_series.loc[final_series["week_end"] == target_week_end, "value"]
                if y_true_final.empty or not math.isfinite(float(y_true_final.iloc[0])):
                    continue
                y_true_final = float(y_true_final.iloc[0])

                # Compute final snapshot prediction for stability anchor.
                y_hat_final = _fit_and_forecast(
                    final_series,
                    origin_week_end=origin,
                    horizon_weeks=h,
                    cfg=cfg,
                )
                if y_hat_final is None or not math.isfinite(float(y_hat_final)):
                    continue

                for c in commits:
                    sha = str(c["commit_sha"])
                    ct = pd.Timestamp(c["commit_time_utc"])
                    snap = _series_for_commit(traj, sha=sha, metric=metric)
                    if snap.empty:
                        continue
                    if snap["week_end"].max() < origin:
                        continue

                    y_hat = _fit_and_forecast(
                        snap,
                        origin_week_end=origin,
                        horizon_weeks=h,
                        cfg=cfg,
                    )
                    if y_hat is None or not math.isfinite(float(y_hat)):
                        continue

                    out_pred_rows.append(
                        {
                            "metric": metric,
                            "origin_week_end": origin.date().isoformat(),
                            "horizon_weeks": int(h),
                            "target_week_end": target_week_end.date().isoformat(),
                            "commit_sha": sha,
                            "commit_time_utc": ct.isoformat(),
                            "is_final_commit": bool(sha == final_sha),
                            "y_hat": float(y_hat),
                            "y_hat_final_commit": float(y_hat_final),
                            "y_true_final": float(y_true_final),
                            "abs_diff_to_final_pred": float(abs(float(y_hat) - float(y_hat_final))),
                            "abs_err_to_final_truth": float(abs(float(y_hat) - float(y_true_final))),
                        }
                    )

    out_preds = (REPO_ROOT / str(args.out_preds)).resolve()
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_preds,
        out_pred_rows,
        fieldnames=[
            "metric",
            "origin_week_end",
            "horizon_weeks",
            "target_week_end",
            "commit_sha",
            "commit_time_utc",
            "is_final_commit",
            "y_hat",
            "y_hat_final_commit",
            "y_true_final",
            "abs_diff_to_final_pred",
            "abs_err_to_final_truth",
        ],
    )

    preds = pd.read_csv(out_preds, sep="\t", dtype={"metric": str, "origin_week_end": str, "target_week_end": str, "commit_sha": str, "commit_time_utc": str})
    preds["commit_time_utc"] = pd.to_datetime(preds["commit_time_utc"], errors="coerce", utc=True)
    preds["horizon_weeks"] = pd.to_numeric(preds["horizon_weeks"], errors="coerce").astype(int)
    preds["abs_diff_to_final_pred"] = pd.to_numeric(preds["abs_diff_to_final_pred"], errors="coerce")
    preds = preds.dropna(subset=["commit_time_utc", "abs_diff_to_final_pred"]).copy()

    stability_rows = []
    for (metric, origin_week_end, horizon_weeks), g in preds.groupby(["metric", "origin_week_end", "horizon_weeks"], sort=True):
        g = g.sort_values("commit_time_utc")
        final_pred = float(g["y_hat_final_commit"].iloc[-1])
        tol = max(float(cfg.tol_abs), float(cfg.tol_rel) * abs(final_pred))
        within = (g["abs_diff_to_final_pred"] <= tol).to_numpy(dtype=bool)
        times = g["commit_time_utc"].to_list()
        if len(times) == 0:
            continue
        first_time = times[0]
        stable_time = None
        # earliest index i such that all j>=i are within tolerance
        for i in range(len(within)):
            if within[i] and within[i:].all():
                stable_time = times[i]
                break
        days_to_stable = (stable_time - first_time).days if stable_time is not None else ""
        stability_rows.append(
            {
                "metric": str(metric),
                "origin_week_end": str(origin_week_end),
                "horizon_weeks": int(horizon_weeks),
                "n_commits_evaluated": int(len(g)),
                "final_pred": float(final_pred),
                "tol_used": float(tol),
                "first_commit_time_utc": first_time.isoformat(),
                "stable_commit_time_utc": stable_time.isoformat() if stable_time is not None else "",
                "days_to_forecast_stable": int(days_to_stable) if days_to_stable != "" else "",
                "max_abs_diff_to_final_pred": float(g["abs_diff_to_final_pred"].max()),
            }
        )

    out_stab = (REPO_ROOT / str(args.out_stability)).resolve()
    out_stab.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_stab,
        stability_rows,
        fieldnames=[
            "metric",
            "origin_week_end",
            "horizon_weeks",
            "n_commits_evaluated",
            "final_pred",
            "tol_used",
            "first_commit_time_utc",
            "stable_commit_time_utc",
            "days_to_forecast_stable",
            "max_abs_diff_to_final_pred",
        ],
    )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory": str(Path(args.trajectory)),
        "final_commit_sha": final_sha,
        "metrics": metrics,
        "origins_since": str(args.origins_since),
        "horizons": horizons,
        "cfg": {
            "y_lags": cfg.y_lags,
            "ridge_alpha": cfg.ridge_alpha,
            "train_window_weeks": cfg.train_window_weeks,
            "tol_abs": cfg.tol_abs,
            "tol_rel": cfg.tol_rel,
        },
        "out": {"preds": str(out_preds.relative_to(REPO_ROOT)), "stability": str(out_stab.relative_to(REPO_ROOT))},
        "rows": {"preds": int(len(out_pred_rows)), "stability": int(len(stability_rows))},
    }
    meta_out = args.out_meta or (str(out_stab) + ".meta.json")
    meta_path = (REPO_ROOT / meta_out).resolve() if not str(meta_out).startswith(str(REPO_ROOT)) else Path(meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(str(out_preds))
    print(str(out_stab))
    return 0


def _series_for_commit(traj: pd.DataFrame, *, sha: str, metric: str) -> pd.DataFrame:
    sub = traj[(traj["commit_sha"].astype(str) == str(sha)) & (traj["metric"].astype(str) == str(metric))].copy()
    if sub.empty:
        return sub
    out = sub[["week_end", "value"]].copy()
    out = out.dropna(subset=["week_end", "value"]).copy()
    out = out.sort_values("week_end").drop_duplicates(subset=["week_end"], keep="last")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).copy()
    return out.reset_index(drop=True)


def _fit_and_forecast(series: pd.DataFrame, *, origin_week_end: pd.Timestamp, horizon_weeks: int, cfg: Cfg) -> float | None:
    s = series.copy()
    s["week_end"] = pd.to_datetime(s["week_end"], errors="coerce")
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    s = s.dropna(subset=["week_end", "value"]).sort_values("week_end").reset_index(drop=True)

    # Ensure origin exists.
    if not (s["week_end"] == origin_week_end).any():
        return None

    # Train on a trailing window ending at origin.
    origin_idx = int(s.index[s["week_end"] == origin_week_end][0])
    start_idx = max(0, origin_idx - int(cfg.train_window_weeks))
    s = s.iloc[start_idx : origin_idx + 1].reset_index(drop=True)
    origin_idx = int(len(s) - 1)

    y = s["value"].to_numpy(dtype=float)
    dt = pd.to_datetime(s["week_end"])
    if len(y) < cfg.y_lags + horizon_weeks + 2:
        return None

    # Feature time i predicts target i+horizon (horizon-specific model).
    max_lag = cfg.y_lags - 1
    rows = []
    targets = []
    for i in range(max_lag, origin_idx - horizon_weeks + 1):
        feat = [float(y[i - j]) for j in range(cfg.y_lags)]
        week = int(dt.iloc[i].isocalendar().week)
        ang = 2.0 * math.pi * (week / 53.0)
        feat.append(float(math.sin(ang)))
        feat.append(float(math.cos(ang)))
        yt = float(y[i + horizon_weeks])
        if not (math.isfinite(yt) and np.isfinite(feat).all()):
            continue
        rows.append(feat)
        targets.append(yt)

    if len(targets) < max(6, cfg.y_lags * 3):
        return None

    X = np.asarray(rows, dtype=float)
    y_next = np.asarray(targets, dtype=float)

    model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", Ridge(alpha=float(cfg.ridge_alpha)))])
    model.fit(X, y_next)

    # Predict from origin.
    feat_o = [float(y[origin_idx - j]) for j in range(cfg.y_lags)]
    week_o = int(dt.iloc[origin_idx].isocalendar().week)
    ang_o = 2.0 * math.pi * (week_o / 53.0)
    feat_o.append(float(math.sin(ang_o)))
    feat_o.append(float(math.cos(ang_o)))
    if not np.isfinite(feat_o).all():
        return None
    y_hat = float(model.predict(np.asarray([feat_o], dtype=float))[0])
    return y_hat


def _write_tsv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())
