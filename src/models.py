# src/models.py
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List
import numpy as np
import pandas as pd


# ---------- metrics ----------
def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    if len(y_true) == 0:
        return {"rmse": np.nan, "mape": np.nan, "smape": np.nan}
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(1e-9, np.abs(y_true))) * 100.0)
    smape = float(np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100.0)
    return {"rmse": rmse, "mape": mape, "smape": smape}


# ---------- baselines ----------
def seasonal_naive(ts: pd.Series, horizon: int) -> pd.Series:
    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")
    if len(ts) >= 12:
        ref = ts.iloc[-12]
    else:
        ref = float(ts.tail(min(3, len(ts))).mean())  # nicer than a flat last value
    idx = pd.period_range(ts.index[-1] + 1, periods=horizon, freq="M")
    return pd.Series([ref] * horizon, index=idx)


def drift_forecast(ts: pd.Series, horizon: int) -> pd.Series:
    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")
    n = len(ts)
    if n < 2:
        return seasonal_naive(ts, horizon)
    slope = (ts.iloc[-1] - ts.iloc[0]) / max(1, n - 1)
    start = ts.iloc[-1]
    idx = pd.period_range(ts.index[-1] + 1, periods=horizon, freq="M")
    vals = [start + slope * (i + 1) for i in range(horizon)]
    return pd.Series(vals, index=idx)


def holt_forecast(ts: pd.Series, horizon: int) -> pd.Series:
    # Works well on short histories; falls back to drift if statsmodels missing/too short
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception:
        return drift_forecast(ts, horizon)

    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")
    y = ts.copy()
    y.index = y.index.to_timestamp()

    if len(y) < 4:
        return drift_forecast(ts, horizon)

    fit = ExponentialSmoothing(y, trend="add", seasonal=None, damped_trend=True).fit(
        optimized=True, use_brute=True
    )
    fc = fit.forecast(horizon)
    out = fc.copy()
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out


# ---------- AutoARIMA (robust for short histories) ----------
def arima_forecast(ts: pd.Series, horizon: int, seasonal: bool = True, m: int = 12) -> pd.Series:
    import pmdarima as pm

    y = ts.copy()
    if isinstance(y.index, pd.PeriodIndex):
        y.index = y.index.to_timestamp()

    season_ok = seasonal and (len(ts) >= 2 * m)
    try:
        model = pm.auto_arima(
            y,
            seasonal=season_ok,
            m=(m if season_ok else 1),
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
    except Exception:
        model = pm.auto_arima(
            y, seasonal=False, m=1, stepwise=True, suppress_warnings=True, error_action="ignore"
        )

    fc = model.predict(n_periods=horizon)
    start = (ts.index[-1] + 1) if isinstance(ts.index, pd.PeriodIndex) else pd.Period(ts.index[-1], freq="M") + 1
    idx = pd.period_range(start, periods=horizon, freq="M")
    return pd.Series(fc, index=idx)


# ---------- XGBoost (adaptive lags; safe if xgboost missing) ----------
def xgb_forecast(ts: pd.Series, horizon: int) -> pd.Series:
    try:
        from xgboost import XGBRegressor
    except Exception:
        # Package not installed → safe fallback
        return seasonal_naive(ts, horizon)

    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")

    default_lags: List[int] = [1, 2, 3, 6, 12]
    lags = [l for l in default_lags if l < len(ts)]
    if len(lags) == 0:
        return seasonal_naive(ts, horizon)

    df = pd.DataFrame({"y": ts})
    for l in lags:
        df[f"lag_{l}"] = ts.shift(l)
    df["month_num"] = ts.index.month
    df["year"] = ts.index.year
    df = df.dropna()
    if df.shape[0] < 3:
        return seasonal_naive(ts, horizon)

    # keep training column order stable
    feature_cols = [f"lag_{l}" for l in lags] + ["month_num", "year"]
    X = df[feature_cols].values
    y = df["y"].values

    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, random_state=42
    )
    model.fit(X, y)

def xgb_importance(ts: pd.Series):
    """Return a Series of XGBoost feature importances, or None if unavailable/too short."""
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None

    if not isinstance(ts.index, pd.PeriodIndex):
        ts = ts.to_period("M")

    # only use lags that actually exist
    lags = [l for l in (1, 2, 3, 6, 12) if l < len(ts)]
    if not lags:
        return None

    df = pd.DataFrame({"y": ts})
    for l in lags:
        df[f"lag_{l}"] = ts.shift(l)
    df["month_num"] = ts.index.month
    df["year"] = ts.index.year
    df = df.dropna()
    if len(df) < 3:
        return None

    X = df[[f"lag_{l}" for l in lags] + ["month_num", "year"]]
    y = df["y"].values

    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, random_state=42
    )
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # recursive forecast with same feature order
    history = df.copy()
    preds = []
    for _ in range(horizon):
        next_p = history.index[-1] + 1  # PeriodIndex
        new_feats = {}
        for l in lags:
            new_feats[f"lag_{l}"] = history["y"].iloc[-l] if len(history) >= l else history["y"].iloc[-1]
        new_feats["month_num"] = next_p.month
        new_feats["year"] = next_p.year

        X_next = np.array([[new_feats[c] for c in feature_cols]], dtype=float)
        yhat = float(model.predict(X_next)[0])
        preds.append(yhat)

        # append to history for future lags
        to_add = {"y": yhat}
        for l in lags:
            to_add[f"lag_{l}"] = history["y"].shift(l).iloc[-1] if len(history) >= l else yhat
        to_add["month_num"] = next_p.month
        to_add["year"] = next_p.year
        history.loc[next_p] = to_add  # index is Period(M)

    idx = pd.period_range(ts.index[-1] + 1, periods=horizon, freq="M")
    return pd.Series(preds, index=idx)


# ---------- Prophet (optional) ----------
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


# ---------- Registry ----------
MODEL_FUNCS = {
    "SeasonalNaive": seasonal_naive,
    "Drift": drift_forecast,
    "Holt": holt_forecast,
    "AutoARIMA": arima_forecast,
    "XGBoost": xgb_forecast,
}
