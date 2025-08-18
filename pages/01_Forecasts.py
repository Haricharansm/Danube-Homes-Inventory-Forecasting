# pages/forecasts.py
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_excel, monthly_aggregate
from src.forecasting import build_series, fit_and_forecast, backtest
from src.visualize import plot_actual_forecasts

st.title("📈 Forecasts")

# ---- Runtime capability checks (so the page works on Python 3.13 too) ----
have_pmdarima = True
try:
    import pmdarima  # noqa: F401
except Exception:
    have_pmdarima = False

have_xgb = True
try:
    import xgboost  # noqa: F401
except Exception:
    have_xgb = False

have_prophet = True
try:
    import prophet  # noqa: F401
except Exception:
    have_prophet = False

# ---- Data input -----------------------------------------------------------
default_path = "data/furniture.xlsx"
data_path = st.text_input(
    "Excel data path",
    value=default_path,
    help="Upload your Excel to data/ or provide an absolute path."
)

# Load
try:
    df, meta = load_excel(data_path)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

if not meta["date"] or not meta["value"]:
    st.error("Could not detect date or sales value columns. Ensure your file has Date or Year/Month and 'Sales Val'.")
    st.stop()

# ---- Filters --------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    store_sel = "All"
    group_sel = "All"
    if meta["store"]:
        store_sel = st.selectbox("Store", ["All"] + sorted(df[meta["store"]].dropna().astype(str).unique().tolist()))
    if meta["group"]:
        group_sel = st.selectbox("Category/Group", ["All"] + sorted(df[meta["group"]].dropna().astype(str).unique().tolist()))

mask = pd.Series(True, index=df.index)
if meta["store"] and store_sel != "All":
    mask &= (df[meta["store"]].astype(str) == store_sel)
if meta["group"] and group_sel != "All":
    mask &= (df[meta["group"]].astype(str) == group_sel)

data = df[mask].copy()
date_col = meta["date"]
val_col = meta["value"]

# ---- Aggregate monthly ----------------------------------------------------
monthly = monthly_aggregate(data, date_col, val_col)
if monthly.empty:
    st.info("No data after filters.")
    st.stop()

# ---- Controls -------------------------------------------------------------
st.subheader("Modeling controls")
horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=12, value=3)

models = ["SeasonalNaive"]
if have_pmdarima:
    models.append("AutoARIMA")
if have_xgb:
    models.append("XGBoost")

use_prophet = st.checkbox("Include Prophet (if installed)", value=False and have_prophet)
if use_prophet and have_prophet:
    models.append("Prophet")

# Inform about disabled models
info_msgs = []
if not have_pmdarima:
    info_msgs.append("AutoARIMA disabled (pmdarima not installed on this runtime).")
if not have_xgb:
    info_msgs.append("XGBoost disabled (xgboost not installed).")
if use_prophet and not have_prophet:
    info_msgs.append("Prophet selected but not installed; uncheck or add it to requirements.")
if info_msgs:
    st.info(" ".join(info_msgs))

# ---- Build series & basic info -------------------------------------------
ts = build_series(monthly, val_col)
st.write(f"History: **{len(ts)} months** | Range: **{ts.index.min()} → {ts.index.max()}**")

# ---- Backtest -------------------------------------------------------------
with st.expander("Backtest (expanding window)", expanded=False):
    folds = st.slider("Folds", 1, 5, 3)
    try:
        metrics = backtest(ts, horizon=horizon, models=models, folds=folds)
        mdf = pd.DataFrame(metrics).T[["rmse", "mape", "smape"]]
        st.dataframe(mdf.style.format({"rmse": "{:.0f}", "mape": "{:.2f}%", "smape": "{:.2f}%"}), use_container_width=True)
    except Exception as e:
        st.warning(f"Backtest skipped: {e}")

# ---- Fit & forecast -------------------------------------------------------
try:
    forecasts = fit_and_forecast(ts, horizon=horizon, models=models)
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()

# ---- Plot -----------------------------------------------------------------
fig = plot_actual_forecasts(ts, forecasts)
st.pyplot(fig)

# ---- Table + download -----------------------------------------------------
out = pd.DataFrame({"actual": ts})
for name, fc in forecasts.items():
    out[name] = fc
st.dataframe(out.tail(24).fillna(""), use_container_width=True)

csv = out.reset_index().rename(columns={"index": "month"})
csv["month"] = csv["month"].astype(str)
st.download_button(
    "Download forecasts CSV",
    data=csv.to_csv(index=False),
    file_name="forecasts.csv",
    mime="text/csv"
)
