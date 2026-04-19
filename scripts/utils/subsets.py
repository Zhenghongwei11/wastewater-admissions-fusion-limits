from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubsetAssignment:
    county_fips: str
    subset: str  # high_fidelity|augmented
    reason: str


def read_subset_registry(path: Path) -> dict[str, SubsetAssignment]:
    out: dict[str, SubsetAssignment] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            cf = (row.get("county_fips") or "").strip()
            subset = (row.get("subset") or "").strip()
            if not cf or not subset:
                continue
            out[cf] = SubsetAssignment(cf, subset, (row.get("reason") or "").strip())
    return out

