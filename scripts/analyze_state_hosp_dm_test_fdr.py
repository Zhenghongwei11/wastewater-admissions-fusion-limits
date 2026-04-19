from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    ps = p[mask]
    if ps.size == 0:
        return out
    order = np.argsort(ps)
    q = ps[order] * float(ps.size) / (np.arange(1, ps.size + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out_masked = np.empty_like(ps)
    out_masked[order] = q
    out[mask] = out_masked
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Add Benjamini–Hochberg FDR q-values to the per-series DM test table.")
    ap.add_argument(
        "--dm",
        default="results/metrics/state_hosp_dm_test.per100k.tsv",
        help="Input DM TSV (default: results/metrics/state_hosp_dm_test.per100k.tsv).",
    )
    ap.add_argument(
        "--out",
        default="results/metrics/state_hosp_dm_test_fdr.per100k.tsv",
        help="Output TSV with q-values (default: results/metrics/state_hosp_dm_test_fdr.per100k.tsv).",
    )
    ap.add_argument(
        "--out-summary",
        default="results/metrics/state_hosp_dm_test_fdr_summary.per100k.tsv",
        help="Small summary TSV output (default: results/metrics/state_hosp_dm_test_fdr_summary.per100k.tsv).",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR threshold for summary counts (default: 0.05).")
    args = ap.parse_args()

    src = (REPO_ROOT / str(args.dm)).resolve()
    if not src.exists():
        raise SystemExit(f"missing DM table: {src}")

    df = pd.read_csv(src, sep="\t")
    req = {"dm_p_value_two_sided", "mean_loss_diff_fusion_minus_ar"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise SystemExit(f"DM table missing columns: {', '.join(miss)}")

    p = pd.to_numeric(df["dm_p_value_two_sided"], errors="coerce").to_numpy(dtype=float)
    df["bh_q_value_two_sided"] = _bh_fdr(p)

    out = (REPO_ROOT / str(args.out)).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

    alpha = float(args.alpha)
    q = pd.to_numeric(df["bh_q_value_two_sided"], errors="coerce")
    diff = pd.to_numeric(df["mean_loss_diff_fusion_minus_ar"], errors="coerce")
    sig = q <= alpha
    worse = sig & (diff > 0)
    better = sig & (diff < 0)
    summary_rows = [
        {
            "alpha": alpha,
            "n_tests": int(len(df)),
            "n_significant_any": int(sig.sum()),
            "frac_significant_any": float(sig.mean()),
            "n_significant_fusion_worse": int(worse.sum()),
            "frac_significant_fusion_worse": float(worse.mean()),
            "n_significant_fusion_better": int(better.sum()),
            "frac_significant_fusion_better": float(better.mean()),
            "notes": "Benjamini–Hochberg FDR applied to two-sided DM p-values across all state–pathogen series.",
        }
    ]

    out_s = (REPO_ROOT / str(args.out_summary)).resolve()
    out_s.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_s, summary_rows)

    print(str(out))
    print(str(out_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

