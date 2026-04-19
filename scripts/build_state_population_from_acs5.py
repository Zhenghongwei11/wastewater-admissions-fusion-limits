from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build state population denominators from ACS5 county covariates.")
    ap.add_argument(
        "--acs-csv",
        required=True,
        help="ACS5 county covariates CSV produced by pull_us_county_acs5_national.py (includes B01003_001E).",
    )
    ap.add_argument("--out", default="results/derived/state_population_acs.tsv", help="Output TSV path.")
    args = ap.parse_args()

    acs_path = (REPO_ROOT / str(args.acs_csv)).resolve()
    if not acs_path.exists():
        raise SystemExit(f"missing ACS CSV: {acs_path}")

    pops: dict[str, float] = {}
    with acs_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "county_fips" not in (r.fieldnames or []) or "B01003_001E" not in (r.fieldnames or []):
            raise SystemExit("ACS CSV must include county_fips and B01003_001E (total population).")
        for row in r:
            cf = (row.get("county_fips") or "").strip().zfill(5)
            if len(cf) != 5 or not cf.isdigit():
                continue
            st = _STATE_FIPS_TO_ABBR.get(cf[:2])
            if not st:
                continue
            try:
                v = float((row.get("B01003_001E") or "").strip())
            except ValueError:
                continue
            if v <= 0:
                continue
            pops[st] = pops.get(st, 0.0) + v

    out_path = (REPO_ROOT / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_path,
        [{"geo_level": "state", "geo_id": st, "population": float(pop)} for st, pop in sorted(pops.items())],
        fieldnames=["geo_level", "geo_id", "population"],
    )
    print(str(out_path.relative_to(REPO_ROOT)))
    return 0


def _write_tsv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


_STATE_FIPS_TO_ABBR = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}


if __name__ == "__main__":
    raise SystemExit(main())

