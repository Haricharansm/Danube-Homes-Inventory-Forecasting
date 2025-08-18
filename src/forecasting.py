import pandas as pd
from typing import Dict, List
from .models import MODEL_FUNCS, evaluate_forecast, prophet_forecast

def build_series(monthly_df: pd.DataFrame, value_col: str) -> pd.Series:
    ts = monthly_df.set_index("month")[value_col].astype(float).sort_index()
    # Ensure PeriodIndex monthly
    if not isinstance(ts.index, pd.PeriodIndex):
        ts.index = pd.PeriodIndex(ts.index, freq="M")
    return ts

def fit_and_forecast(ts: pd.Series, horizon: int, models: List[str]) -> Dict[str, pd.Series]:
    forecasts = {}
    for name in models:
        if name == "Prophet":
            try:
                forecasts[name] = prophet_forecast(ts, horizon)
            except Exception:
                continue
        else:
            forecasts[name] = MODEL_FUNCS[name](ts, horizon)
    return forecasts

def backtest(ts: pd.Series, horizon: int, models: List[str], folds: int=3) -> Dict[str, Dict[str, float]]:
    metrics = {m: {"rmse":0.0,"mape":0.0,"smape":0.0,"n":0} for m in models}
    min_train = max(12, len(ts)//2)
    split_points = [len(ts)-horizon*(i+1) for i in range(folds)][::-1]
    split_points = [s for s in split_points if s > min_train]
    for s in split_points:
        train = ts.iloc[:s]
        test = ts.iloc[s:s+horizon]
        fc_all = fit_and_forecast(train, horizon, models)
        for m, fc in fc_all.items():
            met = evaluate_forecast(test, fc)
            for k in ("rmse","mape","smape"):
                if pd.notna(met[k]):
                    metrics[m][k] += met[k]
            metrics[m]["n"] += 1
    for m in models:
        n = max(1, metrics[m]["n"])
        for k in ("rmse","mape","smape"):
            metrics[m][k] = metrics[m][k] / n
    return metrics
