#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


def _compute_thresholds(panel: pd.DataFrame, *, y_col: str, test_weeks: int, q: float) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for (geo_id, pathogen), g in panel.groupby(["geo_id", "pathogen"], sort=True):
        g = g.sort_values("week_end")
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
        if int(np.isfinite(y).sum()) < max(60, test_weeks + 20):
            continue
        # Availability-respecting threshold: compute from pre-holdout history only.
        y_train = y[: max(0, len(y) - int(test_weeks))]
        y_train = y_train[np.isfinite(y_train)]
        if len(y_train) < 30:
            continue
        thr = float(np.quantile(y_train, q))
        out_rows.append({"geo_id": str(geo_id), "pathogen": str(pathogen), "threshold_quantile": float(q), "threshold_value": thr})
    return pd.DataFrame(out_rows)


def _confusion_metrics(y_true: np.ndarray, y_hat: np.ndarray, *, thr: float) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yh = np.asarray(y_hat, dtype=float)
    keep = np.isfinite(yt) & np.isfinite(yh)
    if int(keep.sum()) == 0:
        return {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}

    yt = yt[keep]
    yh = yh[keep]
    y_event = yt >= float(thr)
    y_pred = yh >= float(thr)

    tp = float(np.sum(y_event & y_pred))
    fp = float(np.sum((~y_event) & y_pred))
    tn = float(np.sum((~y_event) & (~y_pred)))
    fn = float(np.sum(y_event & (~y_pred)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Alarm-threshold sensitivity analysis for one-week-ahead forecasts: false-alarm and detection metrics at 75th/90th percentile surge thresholds."
    )
    ap.add_argument(
        "--panel",
        default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv",
        help="Geo-matched state panel TSV (default: results/panels/wastewater_hosp_panel.geo_matched_state.tsv).",
    )
    ap.add_argument(
        "--y-col",
        default="hosp_admissions_per_100k",
        help="Target column in the panel used to define thresholds (default: hosp_admissions_per_100k).",
    )
    ap.add_argument("--subset", default="geo_matched_state", help="Subset label (default: geo_matched_state).")
    ap.add_argument("--test-weeks", type=int, default=52, help="Holdout weeks (default: 52).")
    ap.add_argument(
        "--predictions",
        default="results/predictions/state_hosp_predictions.delay0_miss000_rev000.per100k.h1.tsv",
        help="Long-format predictions TSV for baseline cell (default: results/predictions/state_hosp_predictions.delay0_miss000_rev000.per100k.h1.tsv).",
    )
    ap.add_argument("--horizon", type=int, default=1, help="Forecast horizon to evaluate (default: 1).")
    ap.add_argument(
        "--models",
        default="ridge_ar,ridge_fusion",
        help="Comma-separated model_ids to include (default: ridge_ar,ridge_fusion).",
    )
    ap.add_argument(
        "--quantiles",
        default="0.75,0.90",
        help="Comma-separated quantiles for threshold definition (default: 0.75,0.90).",
    )
    ap.add_argument(
        "--out-by-series",
        default="results/metrics/state_hosp_alarm_threshold_sensitivity.by_series.per100k.h1.tsv",
        help="Output TSV with per-series alarm metrics.",
    )
    ap.add_argument(
        "--out-summary",
        default="results/metrics/state_hosp_alarm_threshold_sensitivity.summary.per100k.h1.tsv",
        help="Output TSV with summaries across series.",
    )
    args = ap.parse_args()

    panel_path = (REPO_ROOT / str(args.panel)).resolve()
    pred_path = (REPO_ROOT / str(args.predictions)).resolve()
    if not panel_path.exists():
        raise SystemExit(f"missing panel: {panel_path}")
    if not pred_path.exists():
        raise SystemExit(f"missing predictions: {pred_path}")

    panel = pd.read_csv(panel_path, sep="\t", dtype={"geo_id": str, "pathogen": str, "subset": str})
    if str(args.y_col) not in panel.columns:
        raise SystemExit(f"--y-col not in panel: {args.y_col}")
    panel = panel[panel["subset"].astype(str) == str(args.subset)].copy()
    panel["week_end"] = pd.to_datetime(panel["week_end"], errors="coerce")
    panel = panel.dropna(subset=["week_end"]).copy()

    q_list = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    q_list = [q for q in q_list if 0 < q < 1]
    if not q_list:
        raise SystemExit("--quantiles must include values in (0,1)")

    thrs = pd.concat(
        [
            _compute_thresholds(panel, y_col=str(args.y_col), test_weeks=int(args.test_weeks), q=q)
            for q in sorted(set(q_list))
        ],
        ignore_index=True,
    )

    preds = pd.read_csv(pred_path, sep="\t", dtype={"subset": str, "context": str, "geo_id": str, "pathogen": str, "model_id": str})
    needed = {"subset", "context", "geo_level", "geo_id", "pathogen", "horizon_weeks", "model_id", "week_end", "y_true", "y_hat"}
    miss = sorted(needed - set(preds.columns))
    if miss:
        raise SystemExit(f"predictions missing columns: {', '.join(miss)}")

    preds = preds[(preds["subset"].astype(str) == str(args.subset)) & (preds["geo_level"].astype(str) == "state")].copy()
    preds["horizon_weeks"] = pd.to_numeric(preds["horizon_weeks"], errors="coerce")
    preds = preds[preds["horizon_weeks"] == int(args.horizon)].copy()
    preds["week_end"] = pd.to_datetime(preds["week_end"], errors="coerce")
    preds["y_true"] = pd.to_numeric(preds["y_true"], errors="coerce")
    preds["y_hat"] = pd.to_numeric(preds["y_hat"], errors="coerce")
    preds = preds.dropna(subset=["week_end", "y_true", "y_hat"]).copy()

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    preds = preds[preds["model_id"].isin(models)].copy()

    rows: list[dict[str, object]] = []
    for q in sorted(set(q_list)):
        th_q = thrs[thrs["threshold_quantile"] == float(q)].copy()
        if th_q.empty:
            continue
        key_thr = {(r["geo_id"], r["pathogen"]): float(r["threshold_value"]) for _, r in th_q.iterrows()}

        for (ctx, geo_id, pathogen, model_id), g in preds.groupby(["context", "geo_id", "pathogen", "model_id"], sort=True):
            thr = key_thr.get((str(geo_id), str(pathogen)))
            if thr is None or not np.isfinite(thr):
                continue

            g = g.sort_values("week_end")
            cm = _confusion_metrics(g["y_true"].to_numpy(dtype=float), g["y_hat"].to_numpy(dtype=float), thr=thr)
            tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            fdr = _safe_div(fp, tp + fp)  # 1-precision
            fpr = _safe_div(fp, fp + tn)
            fnr = _safe_div(fn, fn + tp)
            prevalence = _safe_div(tp + fn, tp + fp + tn + fn)

            rows.append(
                {
                    "subset": str(args.subset),
                    "context": str(ctx),
                    "geo_level": "state",
                    "geo_id": str(geo_id),
                    "pathogen": str(pathogen),
                    "horizon_weeks": int(args.horizon),
                    "model_id": str(model_id),
                    "threshold_quantile": float(q),
                    "threshold_value": float(thr),
                    "n_weeks": int(len(g)),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "precision": float(precision) if np.isfinite(precision) else "",
                    "recall": float(recall) if np.isfinite(recall) else "",
                    "fdr": float(fdr) if np.isfinite(fdr) else "",
                    "fpr": float(fpr) if np.isfinite(fpr) else "",
                    "fnr": float(fnr) if np.isfinite(fnr) else "",
                    "prevalence": float(prevalence) if np.isfinite(prevalence) else "",
                }
            )

    out_by = (REPO_ROOT / str(args.out_by_series)).resolve()
    out_by.parent.mkdir(parents=True, exist_ok=True)
    by = pd.DataFrame(rows).sort_values(["threshold_quantile", "context", "pathogen", "geo_id", "model_id"]).reset_index(drop=True)
    by.to_csv(out_by, sep="\t", index=False)

    for col in ["precision", "recall", "fdr", "fpr", "fnr", "prevalence"]:
        if col in by.columns:
            by[col] = pd.to_numeric(by[col], errors="coerce")

    summary = (
        by.groupby(["threshold_quantile", "context", "pathogen", "model_id"], as_index=False)
        .agg(
            n_series=("geo_id", "size"),
            median_precision=("precision", "median"),
            median_recall=("recall", "median"),
            median_fdr=("fdr", "median"),
            median_fpr=("fpr", "median"),
            median_fnr=("fnr", "median"),
            median_prevalence=("prevalence", "median"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_fdr=("fdr", "mean"),
            mean_fpr=("fpr", "mean"),
            mean_fnr=("fnr", "mean"),
        )
        .sort_values(["threshold_quantile", "context", "pathogen", "model_id"])
        .reset_index(drop=True)
    )

    out_s = (REPO_ROOT / str(args.out_summary)).resolve()
    out_s.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_s, sep="\t", index=False)

    print(f"OK alarm_threshold_sensitivity: wrote {out_by.relative_to(REPO_ROOT)} and {out_s.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

