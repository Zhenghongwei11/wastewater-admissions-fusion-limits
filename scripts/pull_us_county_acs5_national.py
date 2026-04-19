from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from utils.checksum import sha256_file
from utils.http import build_session, request_json


REPO_ROOT = Path(__file__).resolve().parents[1]


ACS_VARS_DEFAULT = [
    "NAME",
    "B01003_001E",  # total population
    "B19013_001E",  # median household income
    "B17001_001E",  # poverty universe
    "B17001_002E",  # below poverty
]


@dataclass(frozen=True)
class OutputRecord:
    source_id: str
    endpoint_url: str
    out_path: Path
    sha256: str
    bytes: int
    rows: int
    notes: str


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull national ACS 5-year county covariates (single call) for macro analyses.")
    ap.add_argument("--run-id", default=None, help="Run id (default: YYYY-MM-DD_acs5us01).")
    ap.add_argument("--acs-year", type=int, default=2024, help="ACS 5-year API year (default: 2024).")
    ap.add_argument("--acs-vars", default=",".join(ACS_VARS_DEFAULT), help="Comma-separated ACS variables to fetch.")
    args = ap.parse_args()

    today = date.today()
    run_id = args.run_id or f"{today.isoformat()}_acs5us01"

    out_dir = REPO_ROOT / "data" / "raw" / "us_context" / run_id
    log_dir = REPO_ROOT / "logs" / "ingest" / run_id
    audit_dir = REPO_ROOT / "docs" / "audit_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "requests.jsonl"

    user_agent = os.getenv("USER_AGENT") or "USRespiratoryAtlas/0.1 (contact: lgmoon@qzmc.edu.cn)"
    session = build_session(user_agent=user_agent, app_token=None)

    acs_year = int(args.acs_year)
    variables = [v.strip() for v in str(args.acs_vars).split(",") if v.strip()]
    endpoint_url = f"https://api.census.gov/data/{acs_year}/acs/acs5"

    params = {"get": ",".join(variables), "for": "county:*", "in": "state:*"}
    data = request_json(session, url=endpoint_url, params=params, timeout_seconds=180, retries=2, backoff_seconds=2.0, log_path=log_path)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Unexpected ACS response type: {type(data)}")
    header = data[0]
    if not isinstance(header, list):
        raise RuntimeError("ACS header row is not a list")
    idx = {name: i for i, name in enumerate(header)}

    out_rows: list[dict[str, Any]] = []
    for row in data[1:]:
        if not isinstance(row, list):
            continue
        st = str(row[idx["state"]]).zfill(2)
        co = str(row[idx["county"]]).zfill(3)
        rec: dict[str, Any] = {"state": st, "county": co, "county_fips": st + co}
        for v in variables:
            if v in idx:
                rec[v] = row[idx[v]]
        try:
            pov_univ = float(rec.get("B17001_001E") or 0.0)
            pov_below = float(rec.get("B17001_002E") or 0.0)
            rec["poverty_rate"] = "" if pov_univ <= 0 else round(pov_below / pov_univ, 6)
        except ValueError:
            rec["poverty_rate"] = ""
        out_rows.append(rec)

    out_path = out_dir / f"acs5_{acs_year}_county_covariates.csv"
    fieldnames = ["county_fips", "NAME"] + [v for v in variables if v != "NAME"] + ["poverty_rate", "state", "county"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(out_rows, key=lambda x: str(x.get("county_fips", ""))):
            w.writerow(r)

    outputs = [
        OutputRecord(
            source_id="US_CENSUS_ACS5_US_COUNTY_COVARIATES",
            endpoint_url=endpoint_url,
            out_path=out_path,
            sha256=sha256_file(out_path),
            bytes=out_path.stat().st_size,
            rows=len(out_rows),
            notes=f"acs_year={acs_year}; vars={','.join(variables)}",
        )
    ]

    audit_path = audit_dir / "acs5_us_county_outputs.tsv"
    _write_tsv(
        audit_path,
        [
            {
                "source_id": o.source_id,
                "endpoint_url": o.endpoint_url,
                "relative_out_path": str(o.out_path.relative_to(REPO_ROOT)),
                "sha256": o.sha256,
                "bytes": str(o.bytes),
                "rows": str(o.rows),
                "notes": o.notes,
            }
            for o in outputs
        ],
        fieldnames=["source_id", "endpoint_url", "relative_out_path", "sha256", "bytes", "rows", "notes"],
    )

    _upsert_manifest_row(
        REPO_ROOT / "data" / "manifest.tsv",
        source_id="US_CENSUS_ACS5_US_COUNTY_COVARIATES",
        api_endpoint_or_url=endpoint_url,
        spatial_coverage="United States (counties, national)",
        temporal_coverage=f"ACS {acs_year} 5-year estimates (full snapshot)",
        collection_date=today.isoformat(),
        checksum=sha256_file(out_path),
        license_status="TBD (verify Census terms/metadata)",
        scraper_script_path="scripts/pull_us_county_acs5_national.py",
    )

    print(f"National ACS5 county covariates pull complete: {run_id}")
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


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_id",
        "api_endpoint_or_url",
        "spatial_coverage",
        "temporal_coverage",
        "collection_date",
        "data_version",
        "checksum",
        "license_status",
        "scraper_script_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _upsert_manifest_row(
    manifest_path: Path,
    *,
    source_id: str,
    api_endpoint_or_url: str,
    spatial_coverage: str,
    temporal_coverage: str,
    collection_date: str,
    checksum: str,
    license_status: str,
    scraper_script_path: str,
) -> None:
    rows = _read_manifest(manifest_path)
    by_id = {r.get("source_id", ""): r for r in rows if r.get("source_id")}
    if source_id in by_id:
        r = by_id[source_id]
        r["api_endpoint_or_url"] = api_endpoint_or_url
        r["spatial_coverage"] = spatial_coverage
        r["temporal_coverage"] = temporal_coverage
        r["collection_date"] = collection_date
        r["checksum"] = checksum
        r["scraper_script_path"] = scraper_script_path
        if not (r.get("license_status") or "").strip():
            r["license_status"] = license_status
    else:
        by_id[source_id] = {
            "source_id": source_id,
            "api_endpoint_or_url": api_endpoint_or_url,
            "spatial_coverage": spatial_coverage,
            "temporal_coverage": temporal_coverage,
            "collection_date": collection_date,
            "data_version": "",
            "checksum": checksum,
            "license_status": license_status,
            "scraper_script_path": scraper_script_path,
        }
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in rows:
        sid = (r.get("source_id") or "").strip()
        if not sid:
            continue
        if sid == source_id:
            ordered.append(by_id[source_id])
            seen.add(source_id)
        else:
            ordered.append(r)
            seen.add(sid)
    if source_id not in seen:
        ordered.append(by_id[source_id])
    _write_manifest(manifest_path, ordered)


if __name__ == "__main__":
    raise SystemExit(main())

