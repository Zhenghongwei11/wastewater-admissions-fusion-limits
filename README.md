# Wastewater–Admissions Fusion Limits (Public Reproducibility Package)

This repository provides a **reviewer-facing reproducibility package** for a national evaluation of **state-scale wastewater–admissions fusion forecasting** under realistic reporting constraints.

## What this repo reproduces

From public inputs, this package can regenerate:

- Study overview schematic (Figure 1; workflow and supporting analyses)
- State-scale weekly wastewater × admissions panel (NWSS × NHSN)
- Forecasting benchmarks (admissions-only baseline vs fusion; 1–4 week lead times)
- Sensitivity analyses (reporting constraints; paired tests; stratified summaries)
- Revision-history replay outputs for a public revision-history source (stability-window examples)
- Final figure exports (PDF + PNG) used for the analysis package

## Quick start (Docker; recommended)

```bash
docker compose build
docker compose run --rm repro ./scripts/reproduce_one_click.sh
```

Outputs are written to `results/` and `plots/`.

## Local run (Python)

Python 3.11+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
./scripts/reproduce_one_click.sh
```

## Notes on data access

All inputs used by the default pipeline are publicly accessible (CDC Socrata endpoints, U.S. Census API, and a public revision-history repository). See:

- `docs/DATA_MANIFEST.tsv`

## Optional evidence bundle (screenshots)

For reviewer verification, this repo also includes a small set of dashboard screenshots under `evidence/` with an index in:

- `docs/EVIDENCE_MANIFEST.tsv`

## Provenance

- Figure/table provenance: `docs/FIGURE_PROVENANCE.tsv`
- Statistical decision rules and thresholds: `docs/STATISTICAL_DECISION_RULES.md`
