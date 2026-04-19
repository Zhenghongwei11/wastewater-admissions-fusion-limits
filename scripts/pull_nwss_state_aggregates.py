from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from socrata_fetch import SocrataDataset, socrata_query_to_rows, write_rows_csv
from utils.checksum import sha256_file
from utils.http import build_session


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NwssDataset:
    source_id: str
    dataset_id: str

    @property
    def dataset(self) -> SocrataDataset:
        return SocrataDataset(domain="data.cdc.gov", dataset_id=self.dataset_id)


NWSS_DATASETS = [
    NwssDataset(source_id="CDC_NWSS_COVID_WW", dataset_id="j9g8-acpt"),
    NwssDataset(source_id="CDC_NWSS_FLUA_WW", dataset_id="ymmh-divb"),
    NwssDataset(source_id="CDC_NWSS_RSV_WW", dataset_id="45cq-cw4i"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull state-level daily NWSS aggregates (grouped by state_territory × date × pcr_target).")
    ap.add_argument("--run-id", default=None, help="Run id (default: YYYY-MM-DD_nwss_state01).")
    ap.add_argument(
        "--manifest-source-id",
        default="NWSS_STATE_DAILY_AGG_BUNDLE",
        help="source_id used in data/manifest.tsv for this bundle.",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="Optional ISO date (YYYY-MM-DD) lower bound on sample_collect_date.",
    )
    ap.add_argument("--no-trust-env", action="store_true", help="Disable environment proxy variables for requests.")
    args = ap.parse_args()

    today = date.today()
    run_id = args.run_id or f"{today.isoformat()}_nwss_state01"
    since = str(args.since).strip() if args.since else None
    if since:
        date.fromisoformat(since)

    out_dir = REPO_ROOT / "data" / "raw" / "nwss_state" / run_id
    log_dir = REPO_ROOT / "logs" / "ingest" / run_id
    audit_dir = REPO_ROOT / "docs" / "audit_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "requests.jsonl"

    user_agent = os.getenv("USER_AGENT") or "USRespiratoryAtlas/0.1 (contact: lgmoon@qzmc.edu.cn)"
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    session = build_session(user_agent=user_agent, app_token=app_token)
    session.trust_env = not bool(args.no_trust_env)

    audit_rows: list[dict[str, Any]] = []
    bundle_index_rows: list[dict[str, str]] = []

    for ds in NWSS_DATASETS:
        where = "state_territory is not null AND sample_collect_date is not null AND pcr_target is not null"
        where += " AND pcr_target_avg_conc_lin is not null"
        where += " AND population_served is not null"
        if since:
            where += f" AND sample_collect_date >= '{since}'"

        select = (
            "state_territory, sample_collect_date, pcr_target, "
            "avg(pcr_target_avg_conc_lin) as pcr_target_avg_conc_lin_mean, "
            "avg(pcr_target_flowpop_lin) as pcr_target_flowpop_lin_mean, "
            "count(1) as n_samples, "
            "count_distinct(site) as n_sites, "
            "sum(population_served) as population_served_sum"
        )
        group = "state_territory, sample_collect_date, pcr_target"
        order = "sample_collect_date ASC, state_territory ASC, pcr_target ASC"

        out_path = out_dir / f"nwss_{ds.dataset_id}_state_daily_agg.csv"
        rows = socrata_query_to_rows(
            session,
            dataset=ds.dataset,
            where=where,
            select=select,
            group=group,
            order=order,
            timeout_seconds=120,
            page_size=50000,
            log_path=log_path,
        )
        n = write_rows_csv(rows, out_path=out_path, sort_keys=["sample_collect_date", "state_territory", "pcr_target"])
        sha = sha256_file(out_path)
        audit_rows.append(
            {
                "source_id": args.manifest_source_id,
                "dataset_id": ds.dataset_id,
                "endpoint_url": f"https://data.cdc.gov/resource/{ds.dataset_id}.json",
                "where": where,
                "group": group,
                "select": select,
                "rows": n,
                "relative_out_path": str(out_path.relative_to(REPO_ROOT)),
                "sha256": sha,
                "bytes": str(out_path.stat().st_size),
            }
        )
        bundle_index_rows.append({"relative_path": str(out_path.relative_to(REPO_ROOT)), "sha256": sha})

    audit_path = audit_dir / "nwss_state_daily_agg_outputs.tsv"
    _write_tsv(
        audit_path,
        audit_rows,
        fieldnames=["source_id", "dataset_id", "endpoint_url", "where", "group", "select", "rows", "relative_out_path", "sha256", "bytes"],
    )

    bundle_index_path = out_dir / "bundle_index.tsv"
    _write_tsv(bundle_index_path, bundle_index_rows, fieldnames=["relative_path", "sha256"])

    _upsert_manifest_bundle_row(
        REPO_ROOT / "data" / "manifest.tsv",
        source_id=str(args.manifest_source_id),
        collection_date=today.isoformat(),
        api_endpoint_or_url=";".join([f"https://data.cdc.gov/resource/{d.dataset_id}.json" for d in NWSS_DATASETS]),
        spatial_coverage="United States (state-level daily aggregates)",
        temporal_coverage=f"daily aggregates by state_territory × date × pcr_target (run_id={run_id})",
        checksum=sha256_file(bundle_index_path),
        bundle_index_path=bundle_index_path,
        scraper_script_path="scripts/pull_nwss_state_aggregates.py",
    )

    print(f"NWSS state aggregates complete: {run_id}")
    print(f"- Outputs: {out_dir}")
    print(f"- Logs: {log_dir}")
    print(f"- Audit: {audit_dir}")
    return 0


def _write_tsv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _upsert_manifest_bundle_row(
    manifest_path: Path,
    *,
    source_id: str,
    collection_date: str,
    api_endpoint_or_url: str,
    spatial_coverage: str,
    temporal_coverage: str,
    checksum: str,
    bundle_index_path: Path,
    scraper_script_path: str,
) -> None:
    # Keep this small and deterministic: upsert by source_id (append if missing).
    rows: list[dict[str, str]] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                rows.append({k: (v or "").strip() for k, v in row.items()})
        fieldnames = list(r.fieldnames or [])
    else:
        fieldnames = []

    required_fields = [
        "source_id",
        "api_endpoint_or_url",
        "spatial_coverage",
        "temporal_coverage",
        "collection_date",
        "data_version",
        "checksum",
        "license_status",
        "scraper_script_path",
        "notes",
    ]
    for f in required_fields:
        if f not in fieldnames:
            fieldnames.append(f)

    updated = False
    for row in rows:
        if (row.get("source_id") or "").strip() == source_id:
            row["api_endpoint_or_url"] = api_endpoint_or_url
            row["spatial_coverage"] = spatial_coverage
            row["temporal_coverage"] = temporal_coverage
            row["collection_date"] = collection_date
            row["data_version"] = ""
            row["checksum"] = checksum
            row["license_status"] = row.get("license_status") or "US Government work (verify on landing)"
            row["scraper_script_path"] = scraper_script_path
            row["notes"] = f"bundle_index={bundle_index_path.relative_to(REPO_ROOT)}"
            updated = True
            break

    if not updated:
        rows.append(
            {
                "source_id": source_id,
                "api_endpoint_or_url": api_endpoint_or_url,
                "spatial_coverage": spatial_coverage,
                "temporal_coverage": temporal_coverage,
                "collection_date": collection_date,
                "data_version": "",
                "checksum": checksum,
                "license_status": "US Government work (verify on landing)",
                "scraper_script_path": scraper_script_path,
                "notes": f"bundle_index={bundle_index_path.relative_to(REPO_ROOT)}",
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
