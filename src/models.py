import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict

# --- Baseline ---
def seasonal_naive(ts: pd.Series, horizon: int) -> pd.Series:
    # Expect monthly PeriodIndex; convert if needed
    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")
    if len(ts) >= 12:
        ref = ts.iloc[-12]  # same month last year
    else:
        ref = ts.iloc[-1]   # fallback: last observed value
    idx = pd.period_range(ts.index[-1] + 1, periods=horizon, freq="M")
    return pd.Series([ref] * horizon, index=idx)

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    if len(y_true) == 0:
        return {"rmse": np.nan, "mape": np.nan, "smape": np.nan}
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(1e-9, np.abs(y_true))) * 100.0)
    smape = float(np.mean(2*np.abs(err) / (np.abs(y_true)+np.abs(y_pred)+1e-9)) * 100.0)
    return {"rmse": rmse, "mape": mape, "smape": smape}

# --- Auto ARIMA (robust for short histories) ---
def arima_forecast(ts: pd.Series, horizon: int, seasonal: bool=True, m: int=12) -> pd.Series:
    import pmdarima as pm

    # pmdarima expects DatetimeIndex
    y = ts.copy()
    if isinstance(y.index, pd.PeriodIndex):
        y.index = y.index.to_timestamp()

    # Disable seasonality if the series is too short
    season_ok = seasonal and (len(ts) >= 2 * m)
    try:
        model = pm.auto_arima(
            y,
            seasonal=season_ok,
            m=(m if season_ok else 1),
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
    except Exception:
        # Fallback: try non-seasonal auto_arima
        model = pm.auto_arima(
            y, seasonal=False, m=1, stepwise=True,
            suppress_warnings=True, error_action="ignore"
        )

    fc = model.predict(n_periods=horizon)
    # Build monthly PeriodIndex for horizon
    if isinstance(ts.index, pd.PeriodIndex):
        start = ts.index[-1] + 1
    else:
        start = pd.Period(ts.index[-1], freq="M") + 1
    idx = pd.period_range(start, periods=horizon, freq="M")
    return pd.Series(fc, index=idx)

# --- XGBoost with adaptive lags ---
def xgb_forecast(ts: pd.Series, horizon: int) -> pd.Series:
    from xgboost import XGBRegressor

    # Ensure monthly PeriodIndex
    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")

    # Choose only lags that exist
    default_lags = [1, 2, 3, 6, 12]
    lags = [l for l in default_lags if l < len(ts)]
    if len(lags) == 0:
        # Not enough points to train anything — fallback to naive
        return seasonal_naive(ts, horizon)

    # Build training frame
    df = pd.DataFrame({"y": ts})
    for l in lags:
        df[f"lag_{l}"] = ts.shift(l)
    df["month_num"] = ts.index.month
    df["year"] = ts.index.year
    df = df.dropna()
    if df.shape[0] < 3:
        return seasonal_naive(ts, horizon)

    y = df["y"].values
    X = df.drop(columns=["y"]).values

    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, random_state=42
    )
    model.fit(X, y)

    # Recursive forecast
    history = df.copy()
    preds = []
    for _ in range(horizon):
        next_period = history.index[-1] + 1
        hist_y = history["y"]
        new_row = {f"lag_{l}": (hist_y.iloc[-l] if len(hist_y) >= l else hist_y.iloc[-1]) for l in lags}
        new_row["month_num"] = next_period.month
        new_row["year"] = next_period.year
        X_next = np.array([[new_row[k] for k in sorted(new_row.keys())]], dtype=float)

        # Rebuild column order to match training order
        train_cols = [c for c in history.columns if c != "y"]
        X_next = X_next[:, [sorted(new_row.keys()).index(c) for c in train_cols]]

        yhat = model.predict(X_next)[0]
        preds.append(yhat)
        # append forecast for future lags
        history.loc[next_period, "y"] = yhat
        for l in lags:
            history.loc[next_period, f"lag_{l}"] = history["y"].shift(l).iloc[-1]
        history.loc[next_period, "month_num"] = next_period.month
        history.loc[next_period, "year"] = next_period.year

    idx = pd.period_range(ts.index[-1] + 1, periods=horizon, freq="M")
    return pd.Series(preds, index=idx)

# Prophet remains optional; imported in forecasting when requested
def prophet_forecast(ts: pd.Series, horizon: int) -> pd.Series:
    from prophet import Prophet
    y = ts.copy()
    y_idx = y.index.to_timestamp() if isinstance(y.index, pd.PeriodIndex) else y.index
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
}
