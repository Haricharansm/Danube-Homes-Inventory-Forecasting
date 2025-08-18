import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# --- Baseline ---
def seasonal_naive(ts: pd.Series, horizon: int) -> pd.Series:
    # Expect PeriodIndex('M')
    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")
    if len(ts) >= 12:
        last = ts.iloc[-12]
    else:
        last = ts.iloc[-1]
    idx = pd.period_range(ts.index[-1] + 1, periods=horizon, freq="M")
    return pd.Series([last]*horizon, index=idx)

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    if len(y_true) == 0:
        return {"rmse": np.nan, "mape": np.nan, "smape": np.nan}
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(1e-9, np.abs(y_true))) * 100.0)
    smape = float(np.mean(2*np.abs(err) / (np.abs(y_true)+np.abs(y_pred)+1e-9)) * 100.0)
    return {"rmse": rmse, "mape": mape, "smape": smape}

# --- Auto ARIMA ---
def arima_forecast(ts: pd.Series, horizon: int, seasonal: bool=True, m: int=12) -> pd.Series:
    import pmdarima as pm
    # Ensure DatetimeIndex for pmdarima fit
    y = ts.copy()
    if isinstance(y.index, pd.PeriodIndex):
        y.index = y.index.to_timestamp()
    model = pm.auto_arima(y, seasonal=seasonal, m=m, stepwise=True,
                          suppress_warnings=True, error_action="ignore")
    fc = model.predict(n_periods=horizon)
    idx = pd.period_range(ts.index[-1] + 1 if isinstance(ts.index, pd.PeriodIndex)
                          else pd.Period(ts.index[-1], freq="M")+1,
                          periods=horizon, freq="M")
    return pd.Series(fc, index=idx)

# --- XGBoost with lag features ---
def xgb_forecast(ts: pd.Series, horizon: int) -> pd.Series:
    from xgboost import XGBRegressor
    from .features import add_lag_features
    # Ensure PeriodIndex(M)
    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")
    df = add_lag_features(ts).dropna()
    y = df["y"]
    X = df.drop(columns=["y"])
    model = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, random_state=42
    )
    model.fit(X.values, y.values)

    preds = []
    history = df.copy()
    # Recursive walk-forward using PeriodIndex for months
    for step in range(horizon):
        next_period = history.index[-1] + 1  # next month (Period)
        hist_y = history["y"]
        new_row = {f"lag_{l}": (hist_y.iloc[-l] if len(hist_y) >= l else hist_y.iloc[-1]) for l in [1,2,3,6,12]}
        new_row["month_num"] = next_period.month
        new_row["year"] = next_period.year
        X_next = pd.DataFrame([new_row])
        yhat = model.predict(X_next.values)[0]
        preds.append(yhat)
        # append forecast to history for next-step lags
        history.loc[next_period, "y"] = yhat

    idx = pd.period_range(ts.index[-1] + 1, periods=horizon, freq="M")
    return pd.Series(preds, index=idx)

# --- Prophet (optional) ---
def prophet_forecast(ts: pd.Series, horizon: int) -> pd.Series:
    from prophet import Prophet
    y = ts.copy()
    if isinstance(y.index, pd.PeriodIndex):
        y_idx = y.index.to_timestamp()
    else:
        y_idx = y.index
    df = pd.DataFrame({"ds": y_idx, "y": y.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="MS")
    fc = m.predict(future).set_index("ds").iloc[-horizon:]["yhat"]
    out = fc.copy()
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out

MODEL_FUNCS = {
    "SeasonalNaive": seasonal_naive,
    "AutoARIMA": arima_forecast,
    "XGBoost": xgb_forecast,
    # "Prophet": prophet_forecast,  # wired via UI toggle
}
