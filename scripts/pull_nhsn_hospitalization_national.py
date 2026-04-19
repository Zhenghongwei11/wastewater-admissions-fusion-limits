from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from socrata_fetch import SocrataDataset, socrata_query_to_rows
from utils.checksum import sha256_file
from utils.http import build_session
import csv


REPO_ROOT = Path(__file__).resolve().parents[1]

NHSN_DATASET = SocrataDataset(domain="data.cdc.gov", dataset_id="ua7e-t2fy")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pull national NHSN hospitalization metrics by jurisdiction from CDC Socrata.",
    )
    ap.add_argument("--run-id", default=None, help="Run id (default: YYYY-MM-DD_nhsn_hosp01).")
    ap.add_argument("--no-trust-env", action="store_true", help="Disable environment proxy variables for requests.")
    args = ap.parse_args()

    today = date.today()
    run_id = args.run_id or f"{today.isoformat()}_nhsn_hosp01"

    out_dir = REPO_ROOT / "data" / "raw" / "nhsn_hosp" / run_id
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

    print(f"Pulling NHSN hospitalization data from {NHSN_DATASET.dataset_id}...")
    
    # Selecting core fields for respiratory admissions
    select = (
        "weekendingdate, jurisdiction, "
        "totalconfc19newadmhosprep as covid_admissions, "
        "totalconfflunewadmhosprep as flu_admissions, "
        "totalconfrsvnewadmhosprep as rsv_admissions"
    )
    
    rows = socrata_query_to_rows(
        session,
        dataset=NHSN_DATASET,
        where="weekendingdate is not null",
        select=select,
        order="weekendingdate DESC, jurisdiction ASC",
        timeout_seconds=180,
        page_size=10000,
        log_path=log_path,
    )

    if not rows:
        print("No data retrieved from NHSN.")
        return 1

    out_path = out_dir / "nhsn_hospitalizations_by_jurisdiction.csv"
    _write_csv(out_path, rows)

    # Audit
    audit_path = audit_dir / "nhsn_hosp_outputs.tsv"
    audit_rows = [
        {
            "source_id": "CDC_NHSN_HRD_WEEKLY_JURISDICTION",
            "endpoint_url": f"https://data.cdc.gov/resource/{NHSN_DATASET.dataset_id}.json",
            "relative_out_path": str(out_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(out_path),
            "bytes": str(out_path.stat().st_size),
            "notes": f"Jurisdiction-level weekly hospital admissions; rows={len(rows)}",
        }
    ]
    _write_tsv(
        audit_path,
        audit_rows,
        fieldnames=["source_id", "endpoint_url", "relative_out_path", "sha256", "bytes", "notes"],
    )

    print(f"NHSN pull complete: {run_id}")
    print(f"- Output: {out_path}")
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_tsv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
