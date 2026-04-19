from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunRec:
    y_delay_weeks: int
    feature_missing_frac: float
    feature_revision_frac: float
    benchmark_tsv: Path
    summary_tsv: Path
    ablation_tsv: Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Route A benchmark across a grid of synthetic stress parameters (delay × missingness × revision).")
    ap.add_argument("--panel", default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv", help="Input panel TSV.")
    ap.add_argument("--subset", default="geo_matched_state", help="Subset label (default: geo_matched_state).")
    ap.add_argument("--y-col", default="hosp_admissions_per_100k", help="Admissions target column (default: hosp_admissions_per_100k).")
    ap.add_argument("--context", default="both", choices=["with_current_y", "early_warning", "both"], help="Benchmark context (default: both).")
    ap.add_argument("--horizons", default="1,2,3,4", help="Comma-separated horizons (default: 1,2,3,4).")
    ap.add_argument("--test-weeks", type=int, default=52, help="Holdout weeks per series (default: 52).")
    ap.add_argument("--y-lags", type=int, default=8, help="AR lags available (default: 8). Must exceed max(delay)+1.")
    ap.add_argument("--ww-lags", type=int, default=4, help="Wastewater lags (default: 4).")
    ap.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0).")
    ap.add_argument("--mape-eps", type=float, default=1e-5, help="MAPE epsilon (default: 1e-5).")

    ap.add_argument("--delays", default="0,1,2,3,4", help="Comma-separated delays in weeks (default: 0,1,2,3,4).")
    ap.add_argument("--missing-fracs", default="0,0.05,0.10,0.20", help="Comma-separated admissions feature missingness fractions (default: 0,0.05,0.10,0.20).")
    ap.add_argument("--revision-fracs", default="0,0.10,0.30", help="Comma-separated admissions revision probabilities (default: 0,0.10,0.30).")
    ap.add_argument("--revision-scale", type=float, default=0.2, help="Revision scale (default: 0.2 => +/-20 percent).")
    ap.add_argument("--seed", type=int, default=7, help="Synthetic corruption seed (default: 7).")

    ap.add_argument("--out-dir", default="results/stress/state_hosp_stress_grid/per100k", help="Directory for per-run outputs.")
    ap.add_argument("--out-summary", default="results/stress/state_hosp_stress_grid_summary.per100k.tsv", help="Combined per-stratum summary TSV output.")
    ap.add_argument("--out-overall", default="results/stress/state_hosp_stress_grid_overall.per100k.tsv", help="Aggregate overall TSV output.")
    ap.add_argument("--out-meta", default=None, help="Optional JSON metadata output path (default: <out-summary>.meta.json).")
    args = ap.parse_args()

    delays = sorted({int(x.strip()) for x in str(args.delays).split(",") if x.strip() != "" and int(x.strip()) >= 0})
    missing_fracs = sorted({float(x.strip()) for x in str(args.missing_fracs).split(",") if x.strip() != ""})
    revision_fracs = sorted({float(x.strip()) for x in str(args.revision_fracs).split(",") if x.strip() != ""})
    if not delays or not missing_fracs or not revision_fracs:
        raise SystemExit("delays/missing-fracs/revision-fracs must each contain at least one value")

    out_dir = (REPO_ROOT / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunRec] = []
    for d in delays:
        for m in missing_fracs:
            for r in revision_fracs:
                tag = f"delay{int(d)}_miss{_fmt(m)}_rev{_fmt(r)}"
                bench = out_dir / f"state_hosp_forecast_benchmark.{tag}.tsv"
                summ = out_dir / f"state_hosp_forecast_benchmark_summary.{tag}.tsv"
                abla = out_dir / f"state_hosp_signal_ablation.{tag}.tsv"
                _run_one(
                    panel=str(args.panel),
                    subset=str(args.subset),
                    y_col=str(args.y_col),
                    context=str(args.context),
                    horizons=str(args.horizons),
                    test_weeks=int(args.test_weeks),
                    y_lags=int(args.y_lags),
                    ww_lags=int(args.ww_lags),
                    ridge_alpha=float(args.ridge_alpha),
                    mape_eps=float(args.mape_eps),
                    y_delay_weeks=int(d),
                    feature_missing_frac=float(m),
                    feature_revision_frac=float(r),
                    feature_revision_scale=float(args.revision_scale),
                    feature_seed=int(args.seed),
                    out_benchmark=bench,
                    out_summary=summ,
                    out_ablation=abla,
                )
                runs.append(
                    RunRec(
                        y_delay_weeks=int(d),
                        feature_missing_frac=float(m),
                        feature_revision_frac=float(r),
                        benchmark_tsv=bench,
                        summary_tsv=summ,
                        ablation_tsv=abla,
                    )
                )

    frames = []
    for run in runs:
        df = pd.read_csv(run.summary_tsv, sep="\t")
        df.insert(0, "y_delay_weeks", int(run.y_delay_weeks))
        df.insert(1, "feature_missing_frac", float(run.feature_missing_frac))
        df.insert(2, "feature_revision_frac", float(run.feature_revision_frac))
        df["mean_rmse_ridge_ar"] = pd.to_numeric(df["mean_rmse_ridge_ar"], errors="coerce")
        df["mean_rmse_ridge_fusion"] = pd.to_numeric(df["mean_rmse_ridge_fusion"], errors="coerce")
        df["frac_fusion_better_than_ar"] = pd.to_numeric(df["frac_fusion_better_than_ar"], errors="coerce")
        df["mean_delta_rmse_fusion_minus_ar"] = df["mean_rmse_ridge_fusion"] - df["mean_rmse_ridge_ar"]
        frames.append(df)

    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(out_summary, sep="\t", index=False)

    # Aggregate “overall” summary for quick scans.
    def _agg(g: pd.DataFrame) -> pd.Series:
        g = g.copy()
        g["n_series"] = pd.to_numeric(g.get("n_series"), errors="coerce").fillna(0).astype(int)
        g["mean_delta_rmse_fusion_minus_ar"] = pd.to_numeric(g["mean_delta_rmse_fusion_minus_ar"], errors="coerce")
        g["frac_fusion_better_than_ar"] = pd.to_numeric(g["frac_fusion_better_than_ar"], errors="coerce")
        w = g["n_series"].to_numpy(dtype=float)
        denom = float(w.sum()) if float(w.sum()) > 0 else float("nan")
        weighted_delta = float((g["mean_delta_rmse_fusion_minus_ar"] * w).sum() / denom) if denom == denom else float("nan")
        weighted_win = float((g["frac_fusion_better_than_ar"] * w).sum() / denom) if denom == denom else float("nan")
        return pd.Series(
            {
                "n_strata": int(len(g)),
                "total_series": int(g["n_series"].sum()),
                "mean_delta_rmse_fusion_minus_ar": float(g["mean_delta_rmse_fusion_minus_ar"].mean()),
                "weighted_delta_rmse_fusion_minus_ar": weighted_delta,
                "mean_frac_fusion_better_than_ar": float(g["frac_fusion_better_than_ar"].mean()),
                "weighted_frac_fusion_better_than_ar": weighted_win,
            }
        )

    overall = (
        combined.groupby(["y_delay_weeks", "feature_missing_frac", "feature_revision_frac", "context"], dropna=False)
        .apply(_agg)
        .reset_index()
        .sort_values(["context", "y_delay_weeks", "feature_missing_frac", "feature_revision_frac"])
    )
    out_overall = (REPO_ROOT / str(args.out_overall)).resolve()
    out_overall.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out_overall, sep="\t", index=False)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel": str(args.panel),
        "subset": str(args.subset),
        "y_col": str(args.y_col),
        "context": str(args.context),
        "horizons": str(args.horizons),
        "test_weeks": int(args.test_weeks),
        "y_lags": int(args.y_lags),
        "ww_lags": int(args.ww_lags),
        "ridge_alpha": float(args.ridge_alpha),
        "mape_eps": float(args.mape_eps),
        "grid": {
            "delays": delays,
            "missing_fracs": missing_fracs,
            "revision_fracs": revision_fracs,
            "revision_scale": float(args.revision_scale),
            "seed": int(args.seed),
        },
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
        "out": {
            "combined_summary": str(out_summary.relative_to(REPO_ROOT)),
            "overall": str(out_overall.relative_to(REPO_ROOT)),
        },
        "n_runs": int(len(runs)),
    }
    meta_out = args.out_meta or (str(out_summary) + ".meta.json")
    meta_path = (REPO_ROOT / meta_out).resolve() if not str(meta_out).startswith(str(REPO_ROOT)) else Path(meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(str(out_summary))
    print(str(out_overall))
    return 0


def _fmt(x: float) -> str:
    # 0.1 -> 010, 0.05 -> 005
    return f"{int(round(float(x) * 100)):03d}"


def _run_one(
    *,
    panel: str,
    subset: str,
    y_col: str,
    context: str,
    horizons: str,
    test_weeks: int,
    y_lags: int,
    ww_lags: int,
    ridge_alpha: float,
    mape_eps: float,
    y_delay_weeks: int,
    feature_missing_frac: float,
    feature_revision_frac: float,
    feature_revision_scale: float,
    feature_seed: int,
    out_benchmark: Path,
    out_summary: Path,
    out_ablation: Path,
) -> None:
    cmd = [
        "python3",
        str(REPO_ROOT / "scripts" / "run_state_hosp_forecast_benchmark.py"),
        "--panel",
        str(panel),
        "--subset",
        str(subset),
        "--y-col",
        str(y_col),
        "--context",
        str(context),
        "--horizons",
        str(horizons),
        "--test-weeks",
        str(int(test_weeks)),
        "--y-lags",
        str(int(y_lags)),
        "--ww-lags",
        str(int(ww_lags)),
        "--ridge-alpha",
        str(float(ridge_alpha)),
        "--mape-eps",
        str(float(mape_eps)),
        "--y-delay-weeks",
        str(int(y_delay_weeks)),
        "--feature-missing-frac",
        str(float(feature_missing_frac)),
        "--feature-revision-frac",
        str(float(feature_revision_frac)),
        "--feature-revision-scale",
        str(float(feature_revision_scale)),
        "--feature-seed",
        str(int(feature_seed)),
        "--out-benchmark",
        str(out_benchmark),
        "--out-summary",
        str(out_summary),
        "--out-ablation",
        str(out_ablation),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
