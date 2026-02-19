"""Utility helpers for formatting and display."""

import pandas as pd
from typing import List, Dict


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_eur(value: float, decimals: int = 1) -> str:
    """Format a number as EUR millions."""
    return f"€{value:,.{decimals}f}m"


def dict_list_to_df(data: List[Dict]) -> pd.DataFrame:
    """Convert a list of dicts to a formatted DataFrame."""
    return pd.DataFrame(data)


def traffic_light(value: float, green_threshold: float, amber_threshold: float) -> str:
    """Return a traffic-light emoji based on thresholds."""
    if value >= green_threshold:
        return "🟢"
    elif value >= amber_threshold:
        return "🟡"
    return "🔴"
