from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


_PCR_TARGET_TO_PATHOGEN = {
    "sars-cov-2": "COVID-19",
    "fluav": "Influenza",
    "rsv": "RSV",
}


@dataclass(frozen=True)
class NwssDailyAgg:
    state_territory: str
    sample_collect_date: str
    pcr_target: str
    conc_lin_mean: float
    flowpop_lin_mean: float | None
    n_samples: int
    n_sites: int
    population_served_sum: float | None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build geo-matched state-level weekly panel: NWSS wastewater × NHSN hospital admissions.")
    ap.add_argument(
        "--nwss-state-dir",
        default=None,
        help="Directory containing nwss_*_state_daily_agg.csv files (default: latest under data/raw/nwss_state/).",
    )
    ap.add_argument(
        "--nhsn-dir",
        default=None,
        help="Directory containing nhsn_hospitalizations_by_jurisdiction.csv (default: latest under data/raw/nhsn_hosp/).",
    )
    ap.add_argument(
        "--since-week-end",
        default="2023-01-07",
        help="Lower bound (inclusive) on week_end for the joined panel (YYYY-MM-DD; default: 2023-01-07).",
    )
    ap.add_argument(
        "--state-pop-tsv",
        default="results/derived/state_population_acs.tsv",
        help="State population denominators TSV (geo_id, population). Default: results/derived/state_population_acs.tsv.",
    )
    ap.add_argument(
        "--state-population",
        default="results/national_nwss_svi_acs_county.tsv",
        help="County-level ACS table used to derive state populations (default: results/national_nwss_svi_acs_county.tsv).",
    )
    ap.add_argument(
        "--out-state-pop",
        default="results/derived/state_population_acs.tsv",
        help="Output TSV for derived state population denominators (default: results/derived/state_population_acs.tsv).",
    )
    ap.add_argument(
        "--out",
        default="results/panels/wastewater_hosp_panel.geo_matched_state.tsv",
        help="Output joined TSV.",
    )
    args = ap.parse_args()

    nwss_dir = _resolve_latest_dir(REPO_ROOT / "data" / "raw" / "nwss_state", args.nwss_state_dir)
    nhsn_dir = _resolve_latest_dir(REPO_ROOT / "data" / "raw" / "nhsn_hosp", args.nhsn_dir)
    since_week_end = date.fromisoformat(str(args.since_week_end).strip())

    state_pop_tsv = (REPO_ROOT / str(args.state_pop_tsv)).resolve()
    if state_pop_tsv.exists():
        state_pop = _load_state_population_tsv(state_pop_tsv)
    else:
        state_pop = _load_state_population_from_acs(REPO_ROOT / str(args.state_population))
        _write_state_population(REPO_ROOT / str(args.out_state_pop), state_pop)

    nwss_weekly = _load_nwss_state_weekly(nwss_dir, since_week_end=since_week_end)
    nhsn = _load_nhsn_weekly(nhsn_dir, since_week_end=since_week_end)

    out_rows: list[dict[str, Any]] = []
    for (st, pathogen, week_end), wrec in sorted(nwss_weekly.items()):
        y = nhsn.get((st, pathogen, week_end))
        if y is None:
            continue
        pop = float(state_pop.get(st, 0.0) or 0.0)
        adm = float(y["hosp_admissions"])
        adm_per_100k = (adm / pop * 100000.0) if pop > 0 else ""
        out_rows.append(
            {
                "geo_level": "state",
                "geo_id": st,
                "pathogen": pathogen,
                "week_end": week_end,
                "nwss_conc_mean": wrec["nwss_conc_mean"],
                "nwss_flowpop_mean": wrec.get("nwss_flowpop_mean", ""),
                "nwss_n_samples_sum": wrec["nwss_n_samples_sum"],
                "nwss_population_served_max": wrec.get("nwss_population_served_max", ""),
                "nwss_population_coverage_max": (
                    float(wrec.get("nwss_population_served_max", 0.0) or 0.0) / pop if pop > 0 and float(wrec.get("nwss_population_served_max", 0.0) or 0.0) > 0 else ""
                ),
                "hosp_admissions": adm,
                "hosp_admissions_log1p": math.log1p(adm) if adm >= 0 else "",
                "state_population": pop if pop > 0 else "",
                "hosp_admissions_per_100k": adm_per_100k,
                "hosp_admissions_per_100k_log1p": math.log1p(float(adm_per_100k)) if isinstance(adm_per_100k, float) and adm_per_100k >= 0 else "",
                "hosp_source_id": "CDC_NHSN_HRD_WEEKLY_JURISDICTION",
                "subset": "geo_matched_state",
            }
        )

    out_path = (REPO_ROOT / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_path,
        out_rows,
        fieldnames=[
            "geo_level",
            "geo_id",
            "pathogen",
            "week_end",
            "nwss_conc_mean",
            "nwss_flowpop_mean",
            "nwss_n_samples_sum",
            "nwss_population_served_max",
            "nwss_population_coverage_max",
            "hosp_admissions",
            "hosp_admissions_log1p",
            "state_population",
            "hosp_admissions_per_100k",
            "hosp_admissions_per_100k_log1p",
            "hosp_source_id",
            "subset",
        ],
    )
    print(str(out_path))
    return 0


def _resolve_latest_dir(root: Path, override: str | None) -> Path:
    if override:
        p = (REPO_ROOT / str(override)).resolve() if not str(override).startswith(str(REPO_ROOT)) else Path(override)
        if not p.exists():
            raise FileNotFoundError(f"dir not found: {p}")
        return p
    if not root.exists():
        raise FileNotFoundError(f"missing root dir: {root}")
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"no runs under: {root}")
    return runs[-1]


def _load_nwss_state_weekly(nwss_dir: Path, *, since_week_end: date) -> dict[tuple[str, str, str], dict[str, Any]]:
    # Read daily aggregates from all NWSS datasets and aggregate to weekly (week_end Saturday).
    daily: list[NwssDailyAgg] = []
    for p in sorted(nwss_dir.glob("nwss_*_state_daily_agg.csv")):
        daily.extend(_read_nwss_daily_agg(p))

    # weekly bucket: (state, pathogen, week_end) -> weighted sums
    buckets: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in daily:
        pathogen = _PCR_TARGET_TO_PATHOGEN.get(r.pcr_target)
        if not pathogen:
            continue
        week_end = _to_week_end_saturday(r.sample_collect_date)
        if not week_end:
            continue
        d_we = date.fromisoformat(week_end)
        if d_we < since_week_end:
            continue
        key = (r.state_territory, pathogen, week_end)
        b = buckets.setdefault(
            key,
            {
                "w_conc": 0.0,
                "w_flowpop": 0.0,
                "w": 0.0,
                "n_samples_sum": 0.0,
                "pop_served_max": 0.0,
            },
        )
        w = float(max(int(r.n_samples), 0))
        # If a group has zero samples (should not), fall back to weight=1 to keep series continuity.
        w_eff = w if w > 0 else 1.0
        b["w_conc"] += float(r.conc_lin_mean) * w_eff
        if r.flowpop_lin_mean is not None:
            b["w_flowpop"] += float(r.flowpop_lin_mean) * w_eff
        b["w"] += w_eff
        b["n_samples_sum"] += float(max(int(r.n_samples), 0))
        if r.population_served_sum is not None and float(r.population_served_sum) > b["pop_served_max"]:
            b["pop_served_max"] = float(r.population_served_sum)

    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, b in buckets.items():
        if b["w"] <= 0:
            continue
        conc_mean = b["w_conc"] / b["w"]
        if not math.isfinite(conc_mean):
            continue
        flowpop_mean = (b["w_flowpop"] / b["w"]) if b["w_flowpop"] > 0 else None
        out[key] = {
            "nwss_conc_mean": float(conc_mean),
            "nwss_flowpop_mean": float(flowpop_mean) if flowpop_mean is not None and math.isfinite(flowpop_mean) else "",
            "nwss_n_samples_sum": float(b["n_samples_sum"]),
            "nwss_population_served_max": float(b["pop_served_max"]) if b["pop_served_max"] > 0 else "",
        }
    return out


def _read_nwss_daily_agg(path: Path) -> list[NwssDailyAgg]:
    out: list[NwssDailyAgg] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            st = str(row.get("state_territory") or "").strip().upper()
            d = str(row.get("sample_collect_date") or "").strip().split("T")[0]
            tgt = str(row.get("pcr_target") or "").strip()
            conc = _to_float(row.get("pcr_target_avg_conc_lin_mean") or "")
            flowpop = _to_float(row.get("pcr_target_flowpop_lin_mean") or "")
            n_samples = _to_int(row.get("n_samples") or "")
            n_sites = _to_int(row.get("n_sites") or "")
            pop_sum = _to_float(row.get("population_served_sum") or "")
            if not st or not d or not tgt or conc is None:
                continue
            out.append(NwssDailyAgg(st, d, tgt, float(conc), flowpop, int(n_samples), int(n_sites), pop_sum))
    return out


def _load_nhsn_weekly(nhsn_dir: Path, *, since_week_end: date) -> dict[tuple[str, str, str], dict[str, Any]]:
    p = nhsn_dir / "nhsn_hospitalizations_by_jurisdiction.csv"
    if not p.exists():
        raise FileNotFoundError(f"NHSN file not found: {p}")
    df = pd.read_csv(p)
    required = {"weekendingdate", "jurisdiction", "covid_admissions", "flu_admissions", "rsv_admissions"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"NHSN file missing columns: {', '.join(missing)}")

    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        week_end = str(row["weekendingdate"]).split("T")[0]
        if not week_end:
            continue
        try:
            d = date.fromisoformat(week_end)
        except Exception:
            continue
        if d < since_week_end:
            continue
        st = str(row["jurisdiction"]).strip().upper()
        if not st or len(st) != 2:
            continue
        for pathogen, col in [("COVID-19", "covid_admissions"), ("Influenza", "flu_admissions"), ("RSV", "rsv_admissions")]:
            v = _to_float(row.get(col))
            if v is None:
                continue
            out[(st, pathogen, week_end)] = {"hosp_admissions": float(v)}
    return out


def _to_week_end_saturday(iso_date: str) -> str:
    s = (iso_date or "").strip()
    if not s:
        return ""
    try:
        d = date.fromisoformat(s.split("T")[0])
    except Exception:
        return ""
    # Python weekday: Monday=0 ... Sunday=6. Saturday=5.
    delta = (5 - d.weekday()) % 7
    week_end = d + timedelta(days=delta)
    return week_end.isoformat()


def _to_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(v) -> int:
    if v is None:
        return 0
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _write_tsv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_state_population_from_acs(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"ACS county table not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype={"county_fips": str})
    if "county_fips" not in df.columns or "B01003_001E" not in df.columns:
        raise SystemExit("ACS county table must include county_fips and B01003_001E (total population)")
    pops_by_state: dict[str, float] = {}
    for _, r in df.iterrows():
        cf = str(r.get("county_fips") or "").strip().zfill(5)
        if len(cf) != 5 or not cf.isdigit():
            continue
        st = _STATE_FIPS_TO_ABBR.get(cf[:2])
        if not st:
            continue
        v = _to_float(r.get("B01003_001E"))
        if v is None or v <= 0:
            continue
        pops_by_state[st] = pops_by_state.get(st, 0.0) + float(v)
    return pops_by_state


def _load_state_population_tsv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, sep="\t", dtype={"geo_id": str})
    if "geo_id" not in df.columns or "population" not in df.columns:
        raise SystemExit("state population TSV must include geo_id and population")
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        st = str(r.get("geo_id") or "").strip().upper()
        if not st or len(st) != 2:
            continue
        v = _to_float(r.get("population"))
        if v is None or v <= 0:
            continue
        out[st] = float(v)
    return out


def _write_state_population(path: Path, pops_by_state: dict[str, float]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"geo_level": "state", "geo_id": st, "population": float(pop)} for st, pop in sorted(pops_by_state.items())]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["geo_level", "geo_id", "population"], delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


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
    # Territories with county-equivalent FIPS in ACS tables
    "60": "AS",
    "66": "GU",
    "69": "MP",
    "72": "PR",
    "78": "VI",
}

if __name__ == "__main__":
    raise SystemExit(main())
