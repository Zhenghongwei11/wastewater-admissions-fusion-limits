# Compute plan

This package is designed to run on a typical laptop for the “figure regeneration” path and on a workstation for the full “from raw pulls” path.

## Expected runtime (rough guidance)

- Regenerate figures from included derived tables: minutes.
- Re-pull CDC NWSS/NHSN inputs and rebuild the state-scale pipeline: typically tens of minutes to a few hours depending on network and CDC API responsiveness.
- Revision-history replay requires `git` and will clone a public repository; runtime depends on repo size.

## Disk usage

- The pipeline writes downloaded inputs under `data/raw/` and intermediate logs under `logs/`.
- Large raw tables are not committed to git; they are downloaded on demand.

