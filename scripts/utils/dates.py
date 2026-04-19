from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class DateWindow:
    start_date: date
    end_date: date

    @property
    def iso_start(self) -> str:
        return self.start_date.isoformat()

    @property
    def iso_end(self) -> str:
        return self.end_date.isoformat()


def last_n_weeks_window(*, weeks: int, today: date | None = None) -> DateWindow:
    if weeks <= 0:
        raise ValueError("weeks must be positive")
    end = today or date.today()
    start = end - timedelta(days=weeks * 7)
    return DateWindow(start_date=start, end_date=end)


def iso_datetime_utc_midnight(d: date) -> str:
    # Socrata datetime comparisons accept ISO timestamps; keep it simple and explicit.
    return datetime(d.year, d.month, d.day).isoformat(timespec="milliseconds")

