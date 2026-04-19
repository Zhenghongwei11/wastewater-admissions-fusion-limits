from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mechanism diagnostic: summarize short-lag autocorrelation of state-week admissions (supports baseline strength discussion)."
    )
    ap.add_argument(
        "--panel",
        default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv",
        help="Joined panel TSV (default: results/panels/wastewater_hosp_panel.geo_matched_state.tsv).",
    )
    ap.add_argument("--subset", default="geo_matched_state", help="Subset label to filter (default: geo_matched_state).")
    ap.add_argument(
        "--y-col",
        default="hosp_admissions_per_100k",
        help="Admissions column used for autocorrelation (default: hosp_admissions_per_100k).",
    )
    ap.add_argument("--max-lag", type=int, default=4, help="Max lag (weeks) to compute (default: 4).")
    ap.add_argument("--test-weeks", type=int, default=52, help="Exclude final N weeks to mirror benchmark split (default: 52).")
    ap.add_argument(
        "--out-by-series",
        default="results/diagnostics/state_hosp_admissions_autocorr.by_series.per100k.tsv",
        help="Per-series TSV output.",
    )
    ap.add_argument(
        "--out-summary",
        default="results/diagnostics/state_hosp_admissions_autocorr.summary.per100k.tsv",
        help="Summary TSV output.",
    )
    args = ap.parse_args()

    panel_path = (REPO_ROOT / str(args.panel)).resolve()
    if not panel_path.exists():
        raise SystemExit(f"missing panel: {panel_path}")

    y_col = str(args.y_col).strip()
    df = pd.read_csv(panel_path, sep="\t", dtype={"geo_id": str, "pathogen": str, "subset": str})
    df = df.loc[df["subset"].astype(str) == str(args.subset)].copy()
    if y_col not in df.columns:
        raise SystemExit(f"--y-col not found in panel: {y_col}")

    df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["week_end", y_col]).copy()
    df = df.sort_values(["geo_id", "pathogen", "week_end"])

    max_lag = int(args.max_lag)
    test_weeks = int(args.test_weeks)

    rows: list[dict[str, Any]] = []
    for (geo_id, pathogen), g in df.groupby(["geo_id", "pathogen"], sort=True):
        g = g.sort_values("week_end").reset_index(drop=True)
        y = g[y_col].to_numpy(dtype=float)
        t = pd.to_datetime(g["week_end"]).to_numpy()

        if len(y) <= test_weeks + max_lag + 10:
            continue
        y_train = y[: -test_weeks]
        t_train = t[: -test_weeks]

        for lag in range(1, max_lag + 1):
            a = y_train[lag:]
            b = y_train[: -lag]
            keep = np.isfinite(a) & np.isfinite(b)
            if int(keep.sum()) < 30:
                continue
            if float(np.nanstd(a[keep])) < 1e-12 or float(np.nanstd(b[keep])) < 1e-12:
                continue
            c = float(np.corrcoef(a[keep], b[keep])[0, 1])
            if not math.isfinite(c):
                continue
            rows.append(
                {
                    "subset": str(args.subset),
                    "geo_id": str(geo_id),
                    "pathogen": str(pathogen),
                    "y_col": str(y_col),
                    "lag_weeks": int(lag),
                    "n_pairs": int(keep.sum()),
                    "train_start_week_end": str(pd.Timestamp(t_train[0]).date()),
                    "train_end_week_end": str(pd.Timestamp(t_train[-1]).date()),
                    "corr_y_t_y_t_minus_lag": float(c),
                    "notes": "Pearson corr between admissions_t and admissions_{t-lag} within the pre-holdout training pool.",
                }
            )

    out_by = (REPO_ROOT / str(args.out_by_series)).resolve()
    out_by.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_by, rows)

    out_sum = (REPO_ROOT / str(args.out_summary)).resolve()
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_sum, _summarize(rows))

    print(str(out_by))
    print(str(out_sum))
    return 0


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["corr_y_t_y_t_minus_lag"] = pd.to_numeric(df["corr_y_t_y_t_minus_lag"], errors="coerce")
    out: list[dict[str, Any]] = []
    for (subset, lag), g in df.groupby(["subset", "lag_weeks"], sort=True):
        s = g["corr_y_t_y_t_minus_lag"].dropna()
        out.append(
            {
                "subset": str(subset),
                "lag_weeks": int(lag),
                "n_series": int(g[["geo_id", "pathogen"]].drop_duplicates().shape[0]),
                "median_corr": float(s.median()) if len(s) else float("nan"),
                "p25_corr": float(s.quantile(0.25)) if len(s) else float("nan"),
                "p75_corr": float(s.quantile(0.75)) if len(s) else float("nan"),
            }
        )
    return out


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())

