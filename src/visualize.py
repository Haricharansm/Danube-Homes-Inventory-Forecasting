# src/visualize.py
from typing import Dict, Tuple, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt

def _to_dt_index(idx):
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp()
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    try:
        return pd.to_datetime(idx)
    except Exception:
        return pd.Index(idx)

def _prepare(s: Optional[pd.Series]) -> Optional[pd.Series]:
    """Return a safe copy with a datetime-like index, or None if unusable."""
    if s is None or not isinstance(s, pd.Series):
        return None
    if s.empty or s.dropna().empty:
        return None
    s2 = s.copy()
    s2.index = _to_dt_index(s2.index)
    return s2

def plot_actual_forecasts(
    ts: Optional[pd.Series],
    forecasts: Union[Dict[str, Optional[pd.Series]], Tuple[Dict[str, Optional[pd.Series]], list]],
    title: str = "Actual vs Forecast",
):
    # Allow (forecasts_dict, notes) tuples
    if isinstance(forecasts, tuple) and len(forecasts) > 0 and isinstance(forecasts[0], dict):
        forecasts = forecasts[0]
    forecasts = forecasts or {}

    ts_p = _prepare(ts)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if ts_p is not None:
        ax.plot(ts_p.index, ts_p.values, marker="o", linewidth=2, label="Actual")

    plotted = False
    for name, fc in forecasts.items():
        fc_p = _prepare(fc)
        if fc_p is None:
            continue
        ax.plot(fc_p.index, fc_p.values, linestyle="--", marker="o", label=name)
        plotted = True

    if not plotted and ts_p is None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(getattr(ts, "name", "Value") or "Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
