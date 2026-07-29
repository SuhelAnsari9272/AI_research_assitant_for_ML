
from __future__ import annotations

from typing import Optional, Union

Number = Union[int, float]

def format_bytes(num_bytes: Union[int, float]) -> str:
    """Convert a byte count into a human readable string (e.g. '1.4 MB')."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value):,} {unit}"
        value /= 1024.0
    return f"{value:,.1f} PB"

def format_number(value: Optional[Number], decimals: int = 2) -> str:
    """Format a numeric stat, tolerating None and very large integers."""
    if value is None:
        return "N/A"
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"

def format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a 0-100 float as a percentage string, tolerating None."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"

def truncate(text: str, max_len: int = 40) -> str:
    """Truncate long strings (e.g. sample values) for compact display."""
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"