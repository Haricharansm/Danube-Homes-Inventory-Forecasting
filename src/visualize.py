# src/visualize.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union

def _to_datetime_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    idx = s.index
    if isinstance(idx, pd.PeriodIndex):
        s.index = idx.to_timestamp()
    elif isinstance(idx, pd.DatetimeIndex):
        pass  # already OK
    else:
        # try best-effort conversion (e.g., strings like "2025-07")
        try:
            s.index = pd.to_datetime(idx)
        except Exception:
            # leave as-is; matplotlib will still plot but x-axis won’t be dates
            pass
    return s

def plot_actual_forecasts(
    ts: pd.Series,
    forecasts: Union[Dict[str, pd.Series], Tuple[Dict[str, pd.Series], list]],
    title: str = "Actual vs Forecast",
):
    # Allow (forecasts, notes) tuples
    if not isinstance(forecasts, dict) and isinstance(forecasts, tuple) and len(forecasts) >= 1 and isinstance(forecasts[0], dict):
        forecasts = forecasts[0]

    ts_plot = _to_datetime_series(ts)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ts_plot.index, ts_plot.values, label="Actual", linewidth=2, marker="o")

    for name, fc in forecasts.items():
        fc_plot = _to_datetime_series(fc)
        ax.plot(fc_plot.index, fc_plot.values, linestyle="--", marker="o", label=name)

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ts.name or "Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
