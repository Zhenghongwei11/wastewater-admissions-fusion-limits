from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from utils.http import request_json


@dataclass(frozen=True)
class SocrataDataset:
    domain: str
    dataset_id: str

    @property
    def resource_json_url(self) -> str:
        return f"https://{self.domain}/resource/{self.dataset_id}.json"

    @property
    def download_csv_url(self) -> str:
        return f"https://{self.domain}/api/views/{self.dataset_id}/rows.csv?accessType=DOWNLOAD"


def socrata_query_to_rows(
    session: requests.Session,
    *,
    dataset: SocrataDataset,
    where: str | None,
    select: str | None = None,
    group: str | None = None,
    order: str | None = None,
    timeout_seconds: int = 60,
    page_size: int = 50000,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params: dict[str, Any] = {"$limit": page_size, "$offset": offset}
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select
        if group:
            params["$group"] = group
        if order:
            params["$order"] = order

        page = request_json(
            session,
            url=dataset.resource_json_url,
            params=params,
            timeout_seconds=timeout_seconds,
            retries=2,
            backoff_seconds=3.0,
            log_path=log_path,
        )
        if not isinstance(page, list):
            raise TypeError(f"Expected list rows from {dataset.resource_json_url}, got {type(page)}")
        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def write_rows_csv(rows: list[dict[str, Any]], *, out_path: Path, sort_keys: list[str] | None = None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return 0

    if sort_keys:
        rows = sorted(rows, key=lambda r: tuple(str(r.get(k, "")) for k in sort_keys))

    fieldnames: list[str] = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)
