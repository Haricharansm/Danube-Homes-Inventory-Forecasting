import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["year"] = dt.dt.year
    out["month_num"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["is_qtr_start"] = dt.dt.is_quarter_start.astype(int)
    out["is_qtr_end"] = dt.dt.is_quarter_end.astype(int)
    return out

def add_lag_features(ts: pd.Series, lags=(1,2,3,6,12)) -> pd.DataFrame:
    df = pd.DataFrame({"y": ts})
    for l in lags:
        df[f"lag_{l}"] = ts.shift(l)
    df["month_num"] = ts.index.month
    df["year"] = ts.index.year
    return df

def train_val_split_time(df: pd.DataFrame, val_periods: int):
    if val_periods <= 0 or val_periods >= len(df):
        return df, None
    return df.iloc[:-val_periods].copy(), df.iloc[-val_periods:].copy()
