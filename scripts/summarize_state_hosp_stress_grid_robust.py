from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Thresholds:
    min_strata: int
    min_total_series: int


def _fmt_params(r: pd.Series) -> str:
    return f"delay={int(r['y_delay_weeks'])},miss={float(r['feature_missing_frac']):g},rev={float(r['feature_revision_frac']):g}"


def _pick_row(df: pd.DataFrame, *, sort_col: str, ascending: bool) -> dict:
    if df.empty:
        return {"params": "", sort_col: float("nan")}
    r = df.sort_values(sort_col, ascending=ascending).iloc[0]
    return {"params": _fmt_params(r), sort_col: float(r[sort_col])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Route A stress grid overall table with a robust-cells filter (reviewer-safe).")
    ap.add_argument(
        "--overall",
        default="results/stress/state_hosp_stress_grid_overall.per100k.tsv",
        help="Stress grid overall TSV (default: results/stress/state_hosp_stress_grid_overall.per100k.tsv).",
    )
    ap.add_argument("--min-strata", type=int, default=12, help="Minimum strata coverage for a cell to be considered robust (default: 12).")
    ap.add_argument("--min-total-series", type=int, default=300, help="Minimum total series evaluated for a cell to be considered robust (default: 300).")
    ap.add_argument(
        "--out-cells",
        default="results/stress/state_hosp_stress_grid_robust_cells.per100k.tsv",
        help="Robust-cells TSV output.",
    )
    ap.add_argument(
        "--out-summary",
        default="results/stress/state_hosp_stress_grid_robust_summary.per100k.tsv",
        help="Robust summary TSV output.",
    )
    ap.add_argument(
        "--out-note",
        default="docs/ROUTE_A_STRESS_GRID_ROBUST_NOTE.md",
        help="Internal markdown note output.",
    )
    args = ap.parse_args()

    thresholds = Thresholds(min_strata=int(args.min_strata), min_total_series=int(args.min_total_series))

    overall_path = (REPO_ROOT / str(args.overall)).resolve()
    df = pd.read_csv(overall_path, sep="\t")

    needed = [
        "y_delay_weeks",
        "feature_missing_frac",
        "feature_revision_frac",
        "context",
        "n_strata",
        "total_series",
        "weighted_delta_rmse_fusion_minus_ar",
        "weighted_frac_fusion_better_than_ar",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"overall table missing columns: {missing}")

    df = df.copy()
    df["n_strata"] = pd.to_numeric(df["n_strata"], errors="coerce")
    df["total_series"] = pd.to_numeric(df["total_series"], errors="coerce")
    df["weighted_delta_rmse_fusion_minus_ar"] = pd.to_numeric(df["weighted_delta_rmse_fusion_minus_ar"], errors="coerce")
    df["weighted_frac_fusion_better_than_ar"] = pd.to_numeric(df["weighted_frac_fusion_better_than_ar"], errors="coerce")

    df["is_robust"] = (df["n_strata"] >= thresholds.min_strata) & (df["total_series"] >= thresholds.min_total_series)

    robust_cells = df[df["is_robust"]].sort_values(["context", "y_delay_weeks", "feature_missing_frac", "feature_revision_frac"]).reset_index(drop=True)
    out_cells = (REPO_ROOT / str(args.out_cells)).resolve()
    out_cells.parent.mkdir(parents=True, exist_ok=True)
    robust_cells.to_csv(out_cells, sep="\t", index=False)

    rows: list[dict] = []
    note_lines: list[str] = []
    note_lines.append("# Route A Stress Grid (Robust Cells) — Internal Note")
    note_lines.append("")
    note_lines.append(f"- Source: `{overall_path.relative_to(REPO_ROOT)}`")
    note_lines.append(f"- Robust thresholds: `n_strata >= {thresholds.min_strata}` and `total_series >= {thresholds.min_total_series}`")
    note_lines.append("")

    for context, g_all in df.groupby("context"):
        g_all = g_all.copy()
        g_rob = g_all[g_all["is_robust"]].copy()

        baseline = g_all[
            (g_all["y_delay_weeks"] == 0)
            & (g_all["feature_missing_frac"] == 0.0)
            & (g_all["feature_revision_frac"] == 0.0)
        ]
        baseline_row = baseline.iloc[0] if len(baseline) else None

        best_delta = _pick_row(g_rob, sort_col="weighted_delta_rmse_fusion_minus_ar", ascending=True)
        worst_delta = _pick_row(g_rob, sort_col="weighted_delta_rmse_fusion_minus_ar", ascending=False)
        best_win = _pick_row(g_rob, sort_col="weighted_frac_fusion_better_than_ar", ascending=False)

        any_help = bool((g_rob["weighted_delta_rmse_fusion_minus_ar"] < 0).any()) if not g_rob.empty else False

        rows.append(
            {
                "context": context,
                "n_cells_total": int(len(g_all)),
                "n_cells_robust": int(len(g_rob)),
                "baseline_weighted_delta_rmse_fusion_minus_ar": float(baseline_row["weighted_delta_rmse_fusion_minus_ar"]) if baseline_row is not None else float("nan"),
                "baseline_weighted_winrate_fusion_better": float(baseline_row["weighted_frac_fusion_better_than_ar"]) if baseline_row is not None else float("nan"),
                "robust_best_weighted_delta_rmse_fusion_minus_ar": float(best_delta["weighted_delta_rmse_fusion_minus_ar"]),
                "robust_best_delta_params": str(best_delta["params"]),
                "robust_worst_weighted_delta_rmse_fusion_minus_ar": float(worst_delta["weighted_delta_rmse_fusion_minus_ar"]),
                "robust_worst_delta_params": str(worst_delta["params"]),
                "robust_best_weighted_winrate_fusion_better": float(best_win["weighted_frac_fusion_better_than_ar"]),
                "robust_best_winrate_params": str(best_win["params"]),
                "robust_any_cell_fusion_improves_rmse": int(any_help),
            }
        )

        note_lines.append(f"## Context: `{context}`")
        note_lines.append(f"- Robust cells: `{len(g_rob)}` / `{len(g_all)}`")
        if baseline_row is not None:
            note_lines.append(
                "- Baseline cell (delay=0, miss=0, rev=0): "
                f"`weighted_delta={float(baseline_row['weighted_delta_rmse_fusion_minus_ar']):.4f}`, "
                f"`weighted_winrate={float(baseline_row['weighted_frac_fusion_better_than_ar']):.4f}`"
            )
        note_lines.append(
            f"- Best robust delta: `{float(best_delta['weighted_delta_rmse_fusion_minus_ar']):.4f}` at `{best_delta['params']}`"
        )
        note_lines.append(
            f"- Best robust win-rate: `{float(best_win['weighted_frac_fusion_better_than_ar']):.4f}` at `{best_win['params']}`"
        )
        note_lines.append(f"- Any robust cell with delta < 0 (fusion helps RMSE)? `{any_help}`")
        note_lines.append("")

    summary = pd.DataFrame(rows).sort_values(["context"]).reset_index(drop=True)
    out_summary = (REPO_ROOT / str(args.out_summary)).resolve()
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, sep="\t", index=False)

    out_note = (REPO_ROOT / str(args.out_note)).resolve()
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text("\n".join(note_lines).rstrip() + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

