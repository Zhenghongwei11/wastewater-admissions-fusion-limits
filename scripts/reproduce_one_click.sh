#!/usr/bin/env bash
set -euo pipefail

FROM_RAW=0
RUN_STRESS_GRID=0

usage() {
  cat <<'EOF'
Usage: ./scripts/reproduce_one_click.sh [--from-raw] [--stress-grid]

Default (no flags): regenerates plots from the included derived tables under results/.

--from-raw:
  Re-pulls public inputs (CDC NWSS/NHSN, Census ACS; plus NYC revision-history replay),
  rebuilds the main state-scale analysis tables, then regenerates plots.

--stress-grid:
  When used with --from-raw, also reruns the synthetic stress-grid sweep.
  This can take substantially longer.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-raw) FROM_RAW=1 ;;
    --stress-grid) RUN_STRESS_GRID=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$FROM_RAW" -eq 1 ]]; then
  RUN_ID="${RUN_ID:-$(date +%F)_public01}"

  echo "[1/7] Pull CDC NWSS state aggregates"
  python3 scripts/pull_nwss_state_aggregates.py --run-id "${RUN_ID}_nwss_state01" --since 2023-01-01

  echo "[2/7] Pull CDC NHSN jurisdiction admissions"
  python3 scripts/pull_nhsn_hospitalization_national.py --run-id "${RUN_ID}_nhsn_hosp01"

  echo "[3/7] Pull US Census ACS5 county covariates (population)"
  python3 scripts/pull_us_county_acs5_national.py --run-id "${RUN_ID}_acs5us01" --acs-year 2024

  echo "[4/7] Build state population denominators"
  python3 scripts/build_state_population_from_acs5.py \
    --acs-csv "data/raw/us_context/${RUN_ID}_acs5us01/acs5_2024_county_covariates.csv" \
    --out "results/derived/state_population_acs.tsv"

  echo "[5/7] Build state-scale wastewater × admissions panel"
  python3 scripts/build_wastewater_hosp_panel_state.py \
    --nwss-state-dir "data/raw/nwss_state/${RUN_ID}_nwss_state01" \
    --nhsn-dir "data/raw/nhsn_hosp/${RUN_ID}_nhsn_hosp01" \
    --state-pop-tsv "results/derived/state_population_acs.tsv"

  python3 scripts/analyze_wastewater_hosp_leadlag_state.py
  python3 scripts/run_state_hosp_forecast_benchmark.py
  python3 scripts/run_state_hosp_dm_test.py
  python3 scripts/analyze_state_hosp_dm_test_fdr.py
  python3 scripts/analyze_state_hosp_admissions_autocorr.py
  python3 scripts/analyze_state_hosp_turning_point_metrics.py
  python3 scripts/analyze_state_hosp_alarm_threshold_sensitivity.py
  python3 scripts/analyze_state_hosp_fusion_gain_diagnostic_by_lag.py

  if [[ "$RUN_STRESS_GRID" -eq 1 ]]; then
    echo "[6/7] Run synthetic stress-grid sweep (may take time)"
    python3 scripts/run_state_hosp_stress_grid_sweep.py
    python3 scripts/summarize_state_hosp_stress_grid_robust.py
    python3 scripts/summarize_state_hosp_stress_grid_robust_sensitivity.py
    python3 scripts/analyze_state_hosp_svi_stratified_sensitivity.py
    python3 scripts/analyze_state_hosp_stress_grid_robust_svi_stratified.py
    python3 scripts/analyze_state_hosp_coverage_stratified_sensitivity.py
    python3 scripts/analyze_state_hosp_stress_grid_robust_coverage_stratified.py
  fi

  echo "[7/7] NYC revision-history replay (stability-window example)"
  python3 scripts/extract_nyc_git_revision_dynamics.py
  python3 scripts/run_nyc_time_machine_forecast_audit.py
fi

echo "[plots] Regenerate figures"
python3 scripts/make_fig2_state_ww_hosp_leadlag.py --out-prefix plots/fig2_state_ww_hosp_leadlag
python3 scripts/make_fig4_state_hosp_forecast_benchmark.py --out-prefix plots/fig4_state_hosp_forecast_benchmark
python3 scripts/make_fig5_state_hosp_signal_ablation.py --out-prefix plots/fig5_state_hosp_signal_ablation
python3 scripts/make_fig6_state_hosp_asof_stability.py --out-prefix plots/fig6_state_hosp_asof_stability_heatmap
python3 scripts/make_fig_state_hosp_stress_grid_heatmap.py --out-prefix plots/figS_state_hosp_stress_grid_heatmap_rev30
python3 scripts/make_fig_nyc_revision_dynamics.py --out-prefix plots/fig_nyc_revision_dynamics
python3 scripts/make_fig_nyc_forecast_stability.py --out-prefix plots/fig_nyc_forecast_stability
python3 scripts/make_figS1_state_hosp_fusion_gain_diagnostic_by_lag.py --out-prefix plots/figS1_state_hosp_lag_vs_gain
python3 scripts/make_figS2_high_fidelity_ed_forecast_benchmark.py --out-prefix plots/figS2_high_fidelity_ed_benchmark

echo "OK: plots written under plots/"
