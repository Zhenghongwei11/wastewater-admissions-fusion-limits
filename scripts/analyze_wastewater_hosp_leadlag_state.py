from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute lead/lag correlation sweep between state-level NWSS wastewater and NHSN hospital admissions.")
    ap.add_argument(
        "--panel",
        default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv",
        help="Joined panel TSV (output of build_wastewater_hosp_panel_state.py).",
    )
    ap.add_argument("--subset", default="geo_matched_state", help="Subset label to filter (default: geo_matched_state).")
    ap.add_argument("--out", default="results/leadlag/wastewater_hosp_leadlag.geo_matched_state.tsv", help="Output TSV.")
    ap.add_argument("--max-lag", type=int, default=6, help="Max lag in weeks to scan (default: 6).")
    args = ap.parse_args()

    panel_path = REPO_ROOT / str(args.panel)
    rows = _read_tsv(panel_path)
    if not rows:
        raise RuntimeError(f"Empty panel: {panel_path}")

    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        if (r.get("subset") or "").strip() != str(args.subset):
            continue
        geo = (r.get("geo_id") or "").strip()
        path = (r.get("pathogen") or "").strip()
        if not geo or not path:
            continue
        by_key.setdefault((geo, path), []).append(r)

    out_rows: list[dict[str, Any]] = []
    for (geo, path), rr in sorted(by_key.items()):
        rr = sorted(rr, key=lambda x: (x.get("week_end") or ""))
        weeks = [r.get("week_end") or "" for r in rr]
        ww = [_safe_log10p(r.get("nwss_conc_mean") or "") for r in rr]
        y = [_to_float(r.get("hosp_admissions") or "") for r in rr]
        series = [(w, a, b) for (w, a, b) in zip(weeks, ww, y, strict=True) if w and a is not None and b is not None]
        if len(series) < 30:
            continue

        ww_arr = _z(np.asarray([a for (_, a, _) in series], dtype=float))
        y_arr = _z(np.asarray([b for (_, _, b) in series], dtype=float))

        best_lag = None
        best_corr = None
        best_n = None
        for lag in range(-int(args.max_lag), int(args.max_lag) + 1):
            corr, n = _corr_at_lag(ww_arr, y_arr, lag)
            if corr is None:
                continue
            if best_corr is None or corr > best_corr:
                best_corr = corr
                best_lag = lag
                best_n = n

        if best_lag is None or best_corr is None or best_n is None:
            continue

        out_rows.append(
            {
                "geo_level": "state",
                "geo_id": geo,
                "pathogen": path,
                "best_lag_weeks": int(best_lag),
                "corr_at_best": float(best_corr),
                "n_pairs_at_best": int(best_n),
                "notes": "Lag definition: corr( z(log10(1+NWSS)_{t-lag}), z(NHSN admissions_t) ); positive lag => wastewater leads admissions.",
            }
        )

    out_path = (REPO_ROOT / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_path,
        out_rows,
        fieldnames=["geo_level", "geo_id", "pathogen", "best_lag_weeks", "corr_at_best", "n_pairs_at_best", "notes"],
    )
    print(str(out_path))
    return 0


def _corr_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, int] | tuple[None, int]:
    if lag == 0:
        xx = x
        yy = y
    elif lag > 0:
        xx = x[:-lag]
        yy = y[lag:]
    else:
        k = abs(lag)
        xx = x[k:]
        yy = y[:-k]
    n = int(min(len(xx), len(yy)))
    if n < 30:
        return None, n
    c = float(np.corrcoef(xx[:n], yy[:n])[0, 1])
    if not math.isfinite(c):
        return None, n
    return c, n


def _z(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 0:
        return x * 0.0
    return (x - mu) / sd


def _safe_log10p(v: str) -> float | None:
    x = _to_float(v)
    if x is None:
        return None
    if x < 0:
        return None
    return float(math.log10(1.0 + x))


def _to_float(s: str) -> float | None:
    v = (s or "").strip()
    if v == "" or v.lower() == "nan":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return [{k: (v or "").strip() for k, v in row.items()} for row in r]


def _write_tsv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())

