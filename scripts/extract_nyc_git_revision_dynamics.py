from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommitRec:
    sha: str
    commit_time_iso: str


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract NYC respiratory-illness-data git revision dynamics for a single CSV file.")
    ap.add_argument("--repo-url", default="https://github.com/nychealth/respiratory-illness-data.git", help="Git repo URL.")
    ap.add_argument(
        "--repo-dir",
        default="data/external/nychealth_respiratory-illness-data",
        help="Local clone directory (default: data/external/nychealth_respiratory-illness-data).",
    )
    ap.add_argument(
        "--file-path",
        default="data/ED_data_respiratory_illness.csv",
        help="Path within the git repo to analyze (default: data/ED_data_respiratory_illness.csv).",
    )
    ap.add_argument(
        "--since",
        default="2026-01-01",
        help="Only include commits since this date/time (git --since; default: 2026-01-01).",
    )
    ap.add_argument(
        "--out-trajectory",
        default="results/revision/nyc_ed_respiratory_illness_git_trajectory.tsv",
        help="Long table output: commit × week_end × metric (default: results/revision/nyc_ed_respiratory_illness_git_trajectory.tsv).",
    )
    ap.add_argument(
        "--out-summary",
        default="results/revision/nyc_ed_respiratory_illness_git_revision_summary.tsv",
        help="Summary output: per week_end revision counts (default: results/revision/nyc_ed_respiratory_illness_git_revision_summary.tsv).",
    )
    ap.add_argument("--out-meta", default=None, help="Optional JSON metadata output path (default: <out-summary>.meta.json).")
    args = ap.parse_args()

    repo_dir = (REPO_ROOT / str(args.repo_dir)).resolve()
    _ensure_repo(repo_dir, url=str(args.repo_url))

    commits = _list_commits(repo_dir, file_path=str(args.file_path), since=str(args.since))
    if not commits:
        raise SystemExit("no commits found for file in requested window")

    rows = []
    for c in commits:
        csv_text = _git_show(repo_dir, sha=c.sha, file_path=str(args.file_path))
        if csv_text is None:
            continue
        df = _read_csv_text(csv_text)
        if df is None or df.empty:
            continue
        df = _normalize(df)
        for _, r in df.iterrows():
            week_end = str(r["week_end"])
            for metric in [col for col in df.columns if col not in {"week_end"}]:
                v = r.get(metric)
                rows.append(
                    {
                        "commit_sha": c.sha,
                        "commit_time_iso": c.commit_time_iso,
                        "week_end": week_end,
                        "metric": metric,
                        "value": float(v) if pd.notna(v) else "",
                    }
                )

    out_traj = (REPO_ROOT / str(args.out_trajectory)).resolve()
    out_traj.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_traj, rows, fieldnames=["commit_sha", "commit_time_iso", "week_end", "metric", "value"])

    traj = pd.read_csv(out_traj, sep="\t", dtype={"commit_sha": str, "commit_time_iso": str, "week_end": str, "metric": str})
    traj["value"] = pd.to_numeric(traj["value"], errors="coerce")
    traj["commit_time_iso"] = pd.to_datetime(traj["commit_time_iso"], errors="coerce", utc=True)
    traj = traj.dropna(subset=["commit_time_iso", "week_end", "metric"]).copy()

    summaries = []
    for (metric, week_end), g in traj.groupby(["metric", "week_end"], sort=True):
        g = g.sort_values("commit_time_iso")
        n_versions = int(g["value"].nunique(dropna=True))
        n_commits = int(g["commit_sha"].nunique())
        first_seen = g["commit_time_iso"].min()
        last_seen = g["commit_time_iso"].max()
        # "stabilized" := last commit time where value changed.
        last_change = first_seen
        prev = None
        for _, rr in g.iterrows():
            v = rr["value"]
            if prev is None:
                prev = v
                continue
            if pd.notna(v) and pd.notna(prev) and float(v) != float(prev):
                last_change = rr["commit_time_iso"]
            prev = v
        days_to_stable = (last_change - first_seen).days if pd.notna(last_change) and pd.notna(first_seen) else ""

        summaries.append(
            {
                "week_end": week_end,
                "metric": metric,
                "n_commits_touching": n_commits,
                "n_distinct_values": n_versions,
                "first_seen_commit_time_utc": first_seen.isoformat() if pd.notna(first_seen) else "",
                "last_seen_commit_time_utc": last_seen.isoformat() if pd.notna(last_seen) else "",
                "last_change_commit_time_utc": last_change.isoformat() if pd.notna(last_change) else "",
                "days_to_stable": int(days_to_stable) if days_to_stable != "" else "",
            }
        )

    out_sum = (REPO_ROOT / str(args.out_summary)).resolve()
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(
        out_sum,
        summaries,
        fieldnames=[
            "week_end",
            "metric",
            "n_commits_touching",
            "n_distinct_values",
            "first_seen_commit_time_utc",
            "last_seen_commit_time_utc",
            "last_change_commit_time_utc",
            "days_to_stable",
        ],
    )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_url": str(args.repo_url),
        "repo_dir": str(repo_dir.relative_to(REPO_ROOT)),
        "file_path": str(args.file_path),
        "since": str(args.since),
        "n_commits_considered": int(len(commits)),
        "out": {
            "trajectory": str(out_traj.relative_to(REPO_ROOT)),
            "summary": str(out_sum.relative_to(REPO_ROOT)),
        },
    }
    meta_out = args.out_meta or (str(out_sum) + ".meta.json")
    meta_path = (REPO_ROOT / meta_out).resolve() if not str(meta_out).startswith(str(REPO_ROOT)) else Path(meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(str(out_traj))
    print(str(out_sum))
    return 0


def _ensure_repo(repo_dir: Path, *, url: str) -> None:
    if (repo_dir / ".git").exists():
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--all", "--prune"], check=True)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "200", url, str(repo_dir)], check=True)


def _list_commits(repo_dir: Path, *, file_path: str, since: str) -> list[CommitRec]:
    cmd = [
        "git",
        "-C",
        str(repo_dir),
        "log",
        "--since",
        str(since),
        "--pretty=format:%H|%cI",
        "--",
        str(file_path),
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace").strip()
    if not out:
        return []
    recs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|", maxsplit=1)]
        if len(parts) != 2:
            continue
        recs.append(CommitRec(sha=parts[0], commit_time_iso=parts[1]))
    # oldest -> newest (to compute stabilization window).
    return list(reversed(recs))


def _git_show(repo_dir: Path, *, sha: str, file_path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "show", f"{sha}:{file_path}"],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError:
        return None


def _read_csv_text(text: str) -> pd.DataFrame | None:
    # pandas can read from string buffer, but avoid extra deps; use csv module for robust header sniffing.
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        return None
    reader = csv.DictReader(lines)
    rows = list(reader)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Expect a date column (varies across NYC files); normalize to week_end.
    cols = {c.strip(): c for c in df.columns}
    date_col = None
    for cand in ["date", "week_end", "weekendingdate", "week_ending_date"]:
        if cand in cols:
            date_col = cols[cand]
            break
    if not date_col:
        # Fallback: first column.
        date_col = list(df.columns)[0]

    out = df.copy()
    out["week_end"] = pd.to_datetime(out[date_col], errors="coerce").dt.date.astype(str)
    out = out.dropna(subset=["week_end"]).copy()
    out = out.drop(columns=[date_col], errors="ignore")
    # Convert remaining columns to numeric when possible.
    for c in list(out.columns):
        if c == "week_end":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Keep only week_end + numeric columns.
    keep = ["week_end"] + [c for c in out.columns if c != "week_end"]
    return out[keep].sort_values("week_end").reset_index(drop=True)


def _write_tsv(path: Path, rows: list[dict], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())

