# src/forecasting.py
"""
Forecasting utilities used by the Streamlit pages.

Exports:
- build_series(monthly_df, value_col) -> pd.Series (Monthly PeriodIndex)
- fit_and_forecast(ts, horizon, models) -> (dict[name->Series], notes[list[str]])
- backtest(ts, horizon, models, folds=3) -> dict[name -> {rmse,mape,smape,n}]
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .models import MODEL_FUNCS, evaluate_forecast, prophet_forecast


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_month_period_index(idx) -> pd.PeriodIndex:
    """Coerce many index types to a monthly PeriodIndex."""
    if isinstance(idx, pd.PeriodIndex):
        # if freq is None, try to set M; otherwise keep as-is
        return idx.asfreq("M") if (idx.freq is None or idx.freqstr != "M") else idx
    if isinstance(idx, pd.DatetimeIndex):
        return idx.to_period("M")
    # try to parse strings / ints
    try:
        dt = pd.to_datetime(idx)
        return dt.to_period("M")
    except Exception:
        # last resort: build a simple month range 1..n after last point
        return pd.PeriodIndex(idx, freq="M")


def _standardize_forecast(fc: pd.Series, horizon: int, last_index: pd.Period) -> Optional[pd.Series]:
    """Validate and standardize a forecast series (index -> Period[M], length -> horizon)."""
    if fc is None or not isinstance(fc, pd.Series):
        return None
    # coerce numeric
    try:
        fc = pd.to_numeric(fc, errors="coerce")
    except Exception:
        pass
    if fc.dropna().empty:
        return None

    # Coerce to monthly PeriodIndex
    pidx = _to_month_period_index(fc.index)
    fc = pd.Series(fc.values, index=pidx)

    # Ensure it starts right after the last observed timestamp
    expected = pd.period_range(last_index + 1, periods=horizon, freq="M")

    # reindex to expected (trim/pad)
    fc = fc.reindex(expected)
    return fc


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def build_series(monthly_df: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Build a monthly time series with a Period[M] index from a monthly aggregated DataFrame.

    The DataFrame is expected to have a column named 'month' (Period, datetime, or parseable string)
    and a numeric value column `value_col`.
    """
    if "month" not in monthly_df.columns:
        raise ValueError("Expected a 'month' column in the aggregated DataFrame")

    s = monthly_df.set_index("month")[value_col].astype(float).sort_index()
    s.index = _to_month_period_index(s.index)
    s.name = value_col
    return s


def fit_and_forecast(
    ts: pd.Series,
    horizon: int,
    models: List[str],
) -> Tuple[Dict[str, pd.Series], List[str]]:
    """
    Fit each model name in `models` on `ts` and produce an H-step forecast.

    Returns:
      forecasts: dict[name -> pd.Series] with Period[M] index of length `horizon`
      notes: list of info/warning messages (e.g., skipped models)
    """
    forecasts: Dict[str, pd.Series] = {}
    notes: List[str] = []

    # sanitize ts
    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.copy()
        ts.index = _to_month_period_index(ts.index)
    ts = ts.astype(float).sort_index()

    if len(ts) < 2:
        notes.append("Too little history (<2 points); only naive methods may work.")

    for name in models:
        try:
            if name == "Prophet":
                # Prophet is optional; let models.prophet_forecast import Prophet internally
                fc_raw = prophet_forecast(ts, horizon)
            else:
                if name not in MODEL_FUNCS:
                    notes.append(f"{name} not registered; skipped.")
                    continue
                fc_raw = MODEL_FUNCS[name](ts, horizon)

            fc_std = _standardize_forecast(fc_raw, horizon, ts.index[-1])
            if fc_std is None or fc_std.dropna().empty:
                notes.append(f"{name} produced no forecast; skipped.")
                continue

            forecasts[name] = fc_std

        except Exception as e:
            # Never break the app because one model failed
            notes.append(f"{name} skipped: {e}")

    return forecasts, notes


def backtest(
    ts: pd.Series,
    horizon: int,
    models: List[str],
    folds: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Simple expanding-window backtest.

    For each fold, we forecast the next `horizon` months from the training prefix and
    evaluate on that slice. Returns average metrics across folds.
    """
    # Normalize series
    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.copy()
        ts.index = _to_month_period_index(ts.index)
    ts = ts.astype(float).sort_index()

    # Not enough data to backtest robustly
    if len(ts) <= horizon + 2:
        return {m: {"rmse": np.nan, "mape": np.nan, "smape": np.nan, "n": 0} for m in models}

    # Determine split points for expanding window
    # Keep at least ~half of the series for the first train
    min_train = max(6, len(ts) // 2)
    split_points = []
    # create candidate splits from the end, step by horizon
    for i in range(folds):
        s = len(ts) - horizon * (folds - i)
        if s > min_train:
            split_points.append(s)
    if not split_points:
        split_points = [len(ts) - horizon]  # at least one split

    # Initialize metrics
    metrics: Dict[str, Dict[str, float]] = {
        m: {"rmse": 0.0, "mape": 0.0, "smape": 0.0, "n": 0} for m in models
    }

    for s in split_points:
        train = ts.iloc[:s]
        test = ts.iloc[s : s + horizon]
        if test.empty:
            continue

        fdict, _ = fit_and_forecast(train, horizon, models)

        for m in models:
            if m not in fdict:
                continue
            met = evaluate_forecast(test, fdict[m])
            # accumulate only valid numbers
            for k in ("rmse", "mape", "smape"):
                if pd.notna(met.get(k, np.nan)):
                    metrics[m][k] += float(met[k])
            metrics[m]["n"] += 1

    # average
    for m in metrics:
        n = max(1, metrics[m]["n"])
        for k in ("rmse", "mape", "smape"):
            metrics[m][k] = metrics[m][k] / n

    return metrics
