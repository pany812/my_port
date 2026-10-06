"""Minimal markdown table rendering (avoids pandas' optional ``tabulate`` dependency)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

Formatter = Callable[[object], str]


def pct(digits: int = 1) -> Formatter:
    """Decimal -> percent string, e.g. 0.0512 -> '5.1%'. NaN -> '–'. No '-0.0%'."""

    def fmt(v) -> str:
        if pd.isna(v):
            return "–"
        text = f"{float(v) * 100:.{digits}f}"
        return f"{text.lstrip('-') if float(text) == 0 else text}%"

    return fmt


def num(digits: int = 0) -> Formatter:
    return lambda v: "–" if pd.isna(v) else f"{float(v):,.{digits}f}"


def md_table(df: pd.DataFrame, formats: Mapping[str, Formatter] | None = None) -> str:
    """Render ``df`` (columns as headers, index ignored) as a GitHub markdown table."""
    if df.empty:
        return "_(none)_"
    formats = formats or {}
    header = [str(c) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            text = formats[c](v) if c in formats else ("–" if _missing(v) else str(v))
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        rows.append(cells)
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    align = ["---:" if n else ":---" for n in numeric]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _missing(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False
