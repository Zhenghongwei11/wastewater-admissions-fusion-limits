from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from arcgis_fetch import ArcGisLayer, arcgis_query_all, fetch_layer_schema, write_arcgis_csv
from utils.checksum import sha256_file
from utils.http import build_session


REPO_ROOT = Path(__file__).resolve().parents[1]


SVI_2022_COUNTY_LAYER = "https://onemap.cdc.gov/OneMapServices/rest/services/SVI/CDC_ATSDR_Social_Vulnerability_Index_2022_USA/FeatureServer/1"

# Keep this minimal but useful for Figure 1 stratification.
SVI_FIELDS_DEFAULT = [
    "FIPS",
    "ST_ABBR",
    "COUNTY",
    "E_TOTPOP",
    "E_POV150",
    "E_UNEMP",
    "E_NOHSDP",
    "E_UNINSUR",
    "E_MINRTY",
    "RPL_THEMES",
    "RPL_THEME1",
    "RPL_THEME2",
    "RPL_THEME3",
    "RPL_THEME4",
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
    ap = argparse.ArgumentParser(description="Pull CDC/ATSDR SVI 2022 county attributes via ArcGIS FeatureServer.")
    ap.add_argument("--run-id", default=None, help="Run id (default: YYYY-MM-DD_svi2022county01).")
    ap.add_argument("--layer-url", default=SVI_2022_COUNTY_LAYER, help="SVI county layer URL.")
    ap.add_argument("--fields", default=",".join(SVI_FIELDS_DEFAULT), help="Comma-separated list of fields to pull.")
    args = ap.parse_args()

    today = date.today()
    run_id = args.run_id or f"{today.isoformat()}_svi2022county01"

    out_dir = REPO_ROOT / "data" / "raw" / "svi" / run_id
    log_dir = REPO_ROOT / "logs" / "ingest" / run_id
    audit_dir = REPO_ROOT / "docs" / "audit_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "requests.jsonl"

    user_agent = os.getenv("USER_AGENT") or "USRespiratoryAtlas/0.1 (contact: lgmoon@qzmc.edu.cn)"
    session = build_session(user_agent=user_agent, app_token=None)
    session.trust_env = True

    layer_url = str(args.layer_url).strip().rstrip("/")
    layer = ArcGisLayer(layer_url=layer_url)
    schema = fetch_layer_schema(session, layer=layer, timeout_seconds=60, log_path=log_path)
    date_fields = [f["name"] for f in schema.get("fields", []) if f.get("type") == "esriFieldTypeDate"]

    fields = [f.strip() for f in str(args.fields).split(",") if f.strip()]
    out_fields = ",".join(fields) if fields else "*"

    rows = arcgis_query_all(
        session,
        layer=layer,
        where="1=1",
        out_fields=out_fields,
        order_by_fields="FIPS ASC",
        result_record_count=2000,
        timeout_seconds=90,
        log_path=log_path,
    )

    out_path = out_dir / "svi_2022_county.csv"
    n = write_arcgis_csv(rows, out_path=out_path, date_fields_ms=date_fields, sort_keys=["FIPS"])

    # Normalize key join field for downstream merges.
    _normalize_fips_inplace(out_path)

    outputs = [
        OutputRecord(
            source_id="CDC_ATSDR_SVI_2022_COUNTY",
            endpoint_url=layer_url,
            out_path=out_path,
            sha256=sha256_file(out_path),
            bytes=out_path.stat().st_size,
            rows=n,
            notes=f"fields={','.join(fields)}",
        )
    ]

    audit_path = audit_dir / "svi_2022_county_outputs.tsv"
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
        source_id="CDC_ATSDR_SVI_2022_COUNTY",
        api_endpoint_or_url=layer_url,
        spatial_coverage="United States (counties, national)",
        temporal_coverage="SVI 2022 (county; static snapshot)",
        collection_date=today.isoformat(),
        checksum=sha256_file(out_path),
        license_status="TBD (verify CDC/ATSDR SVI terms)",
        scraper_script_path="scripts/pull_cdc_svi_2022_county_arcgis.py",
    )

    print(f"CDC SVI 2022 county pull complete: {run_id}")
    print(f"- Outputs: {out_dir}")
    print(f"- Logs: {log_dir}")
    print(f"- Audit: {audit_dir}")
    return 0


def _normalize_fips_inplace(path: Path) -> None:
    # Ensure FIPS is 5-digit (zero-padded) to match county_fips elsewhere.
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = list(r.fieldnames or [])
    if "FIPS" not in fieldnames:
        return
    for row in rows:
        row["FIPS"] = (row.get("FIPS") or "").strip().zfill(5)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


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

