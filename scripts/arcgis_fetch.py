from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from utils.http import request_json


@dataclass(frozen=True)
class ArcGisLayer:
    layer_url: str  # .../FeatureServer/<layer_id>

    @property
    def query_url(self) -> str:
        return f"{self.layer_url}/query"

    @property
    def schema_url(self) -> str:
        return f"{self.layer_url}"


def fetch_layer_schema(session: requests.Session, *, layer: ArcGisLayer, timeout_seconds: int = 60, log_path: Path | None = None) -> dict[str, Any]:
    return request_json(session, url=layer.schema_url, params={"f": "json"}, timeout_seconds=timeout_seconds, retries=2, backoff_seconds=2.0, log_path=log_path)


def arcgis_query_all(
    session: requests.Session,
    *,
    layer: ArcGisLayer,
    where: str,
    out_fields: str = "*",
    order_by_fields: str | None = None,
    result_record_count: int = 2000,
    timeout_seconds: int = 60,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params: dict[str, Any] = {
            "f": "json",
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": result_record_count,
        }
        if order_by_fields:
            params["orderByFields"] = order_by_fields

        data = request_json(session, url=layer.query_url, params=params, timeout_seconds=timeout_seconds, retries=2, backoff_seconds=2.0, log_path=log_path)
        if isinstance(data, dict) and data.get("error"):
            err = data["error"] or {}
            code = err.get("code", "unknown")
            message = err.get("message", "unknown error")
            details = err.get("details")
            raise RuntimeError(f"ArcGIS query error (code={code}): {message} details={details}")
        feats = data.get("features") or []
        if not feats:
            break
        for f in feats:
            attrs = f.get("attributes") or {}
            all_rows.append(attrs)
        exceeded = bool(data.get("exceededTransferLimit"))
        # ArcGIS servers may enforce a maxRecordCount smaller than result_record_count; in that case
        # they return fewer rows but set exceededTransferLimit=true. Page until the server indicates
        # there are no more results.
        if len(feats) < result_record_count and not exceeded:
            break
        offset += len(feats)
    return all_rows


def write_arcgis_csv(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    date_fields_ms: Iterable[str],
    sort_keys: list[str] | None = None,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return 0

    date_fields = set(date_fields_ms)
    normalized: list[dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        for k in list(rr.keys()):
            if k in date_fields and isinstance(rr.get(k), (int, float)):
                rr[f"{k}_iso"] = datetime.fromtimestamp(rr[k] / 1000, tz=timezone.utc).date().isoformat()
        normalized.append(rr)

    if sort_keys:
        normalized = sorted(normalized, key=lambda r: tuple(str(r.get(k, "")) for k in sort_keys))

    fieldnames = sorted({k for r in normalized for k in r.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in normalized:
            w.writerow(r)
    return len(normalized)
