"""Session-date parsing and relative date resolution for conversational memory."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# 1:56 pm on 8 May, 2023  |  8 May 2023 | May 8, 2023 | 2023-05-08
_LOOSE_DATE = re.compile(
    r"(?:"
    r"(?:\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?"
    r"(?P<d1>\d{1,2})\s+(?P<m1>[A-Za-z]+),?\s+(?P<y1>\d{4})"
    r"|"
    r"(?P<m2>[A-Za-z]+)\s+(?P<d2>\d{1,2}),?\s+(?P<y2>\d{4})"
    r"|"
    r"(?P<y3>\d{4})-(?P<m3>\d{2})-(?P<d3>\d{2})"
    r")"
)


def parse_loose_date(value: str) -> datetime | None:
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    match = _LOOSE_DATE.search(text)
    if not match:
        return None
    if match.group("y1"):
        month = _MONTHS.get(match.group("m1").lower())
        if not month:
            return None
        return datetime(int(match.group("y1")), month, int(match.group("d1")))
    if match.group("y2"):
        month = _MONTHS.get(match.group("m2").lower())
        if not month:
            return None
        return datetime(int(match.group("y2")), month, int(match.group("d2")))
    return datetime(int(match.group("y3")), int(match.group("m3")), int(match.group("d3")))


def format_day_month_year(when: datetime) -> str:
    return f"{when.day} {when.strftime('%B')} {when.year}"


def resolve_relative_dates(text: str, anchor: datetime | None) -> str:
    """Replace yesterday/today/last year/etc. with absolute dates using *anchor*."""
    if anchor is None:
        return text
    out = text
    replacements = [
        (r"\byesterday\b", f"on {format_day_month_year(anchor - timedelta(days=1))}"),
        (r"\btoday\b", f"on {format_day_month_year(anchor)}"),
        (r"\btomorrow\b", f"on {format_day_month_year(anchor + timedelta(days=1))}"),
        (r"\blast\s+year\b", f"in {anchor.year - 1}"),
        (r"\bthis\s+year\b", f"in {anchor.year}"),
        (
            r"\blast\s+month\b",
            f"in {format_day_month_year(anchor.replace(day=1) - timedelta(days=1))}",
        ),
        (r"\ba\s+year\s+ago\b", f"in {anchor.year - 1}"),
        (r"\btwo\s+years\s+ago\b", f"in {anchor.year - 2}"),
    ]
    for pattern, value in replacements:
        out = re.sub(pattern, value, out, flags=re.IGNORECASE)
    return out
