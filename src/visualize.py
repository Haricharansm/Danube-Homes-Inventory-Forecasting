# src/visualize.py
from typing import Dict, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt

def _coerce_series(obj) -> pd.Series | None:
    """Return a copy as Series with a datetime-like index, or None if unusable."""
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        s = obj.copy()
    else:
        try:
            s = pd.Series(obj)
        except Exception:
            return None

    # drop NA-only
    if s.dropna().empty:
        return None

    # normalize index -> DatetimeIndex when possible
    idx = s.index
    try:
        if isinstance(idx, pd.PeriodIndex):
            s.index = idx.to_timestamp()
        elif isinstance(idx, pd.DatetimeIndex):
            pass
        else:
            s.index = pd.to_datetime(idx)
    except Exception:
        # leave as-is; we'll still plot, just without date ticks
        pass
    return s

def plot_actual_forecasts(
    ts: pd.Series,
    forecasts: Union[Dict[str, pd.Series], Tuple[Dict[str, pd.Series], list]],
    title: str = "Actual vs Forecast",
):
    # Allow (forecasts, notes) tuples
    if isinstance(forecasts, tuple) and len(forecasts) > 0 and isinstance(forecasts[0], dict):
        forecasts = forecasts[0]

    forecasts = forecasts or {}
    ts_plot = _coerce_series(ts)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if ts_plot is not None:
        ax.plot(ts_plot.index, ts_plot.values, label="Actual", linewidth=2, marker="o")

    plotted_any = False
    for name, fc in list(forecasts.items()):
        fc_plot = _coerce_series(fc)
        if fc_plot is None:
            continue
        ax.plot(fc_plot.index, fc_plot.values, linestyle="--", marker="o", label=name)
        plotted_any = True

    if not plotted_any and ts_plot is None:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(getattr(ts, "name", "Value") or "Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
