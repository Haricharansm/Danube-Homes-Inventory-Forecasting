import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import load_excel, monthly_aggregate
from src.forecasting import build_series, fit_and_forecast, backtest
from src.visualize import plot_actual_forecasts

@st.cache_data(show_spinner=False)
def _cached_load(path_text: str, upload_bytes: bytes | None):
    from io import BytesIO
    if upload_bytes is not None:
        buf = BytesIO(upload_bytes)
        return load_excel(buf)
    else:
        return load_excel(path_text)

st.title("📈 Forecasts")

# ---- Runtime capability checks ----
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

# ---- Data input ----
default_path = "data/furniture.xlsx"
col1, col2 = st.columns([2, 1], gap="small")
with col1:
    data_path = st.text_input(
        "Excel data path",
        value=default_path,
        help="Path inside repo (e.g., data/furniture.xlsx) or absolute path."
    )
with col2:
    uploaded = st.file_uploader("…or upload file", type=["xlsx", "xls", "csv"])

if uploaded is None:
    p = Path(data_path)
    st.caption(f"Path exists: {p.exists()} | Size: {p.stat().st_size if p.exists() else 'n/a'} bytes")

# ---- Load ----
try:
    if uploaded is not None:
        df, meta = _cached_load(data_path, uploaded.getvalue())
    else:
        df, meta = _cached_load(data_path, None)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()
    
# ---- Filters ----
with st.sidebar:
    st.header("Filters")
    store_sel = "All"
    group_sel = "All"
    if meta["store"]:
        store_sel = st.selectbox(
            "Store",
            ["All"] + sorted(df[meta["store"]].dropna().astype(str).unique().tolist())
        )
    if meta["group"]:
        group_sel = st.selectbox(
            "Category/Group",
            ["All"] + sorted(df[meta["group"]].dropna().astype(str).unique().tolist())
        )

mask = pd.Series(True, index=df.index)
if meta["store"] and store_sel != "All":
    mask &= (df[meta["store"]].astype(str) == store_sel)
if meta["group"] and group_sel != "All":
    mask &= (df[meta["group"]].astype(str) == group_sel)

data = df[mask].copy()
date_col = meta["date"]
val_col = meta["value"]

# ---- Aggregate monthly ----
monthly = monthly_aggregate(data, date_col, val_col)
if monthly.empty:
    st.info("No data after filters.")
    st.stop()
ts = build_series(monthly, val_col)
last = float(ts.iloc[-1])
prev = float(ts.iloc[-2]) if len(ts) >= 2 else last
delta = (last - prev) / prev * 100 if prev != 0 else 0
vol = float(ts.pct_change().dropna().std() * 100) if len(ts) > 2 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Last month", f"{last:,.0f}", f"{delta:+.1f}% vs prev")
c2.metric("3-month total", f"{ts.tail(3).sum():,.0f}")
c3.metric("Volatility (σ of MoM)", f"{vol:.1f}%")
# ---- Controls ----
st.subheader("Modeling controls")
max_h = int(min(12, max(1, len(ts)//2)))
horizon = st.slider("Forecast horizon (months)", 1, max_h, min(3, max_h), help="Max set by your history length")

models = ["SeasonalNaive", "Drift", "Holt"]  # add these in step 2
if have_pmdarima: models.append("AutoARIMA")
if have_xgb: models.append("XGBoost")

use_prophet = st.checkbox("Include Prophet (if installed)", value=False and have_prophet)
if use_prophet and have_prophet: models.append("Prophet")
# ---- Build series ----
ts = build_series(monthly, val_col)
st.write(f"History: **{len(ts)} months** | Range: **{ts.index.min()} → {ts.index.max()}**")

# ---- Backtest ----
with st.expander("Backtest (expanding window)", expanded=False):
    folds = st.slider("Folds", 1, 5, 3)
    try:
        metrics = backtest(ts, horizon=horizon, models=models, folds=folds)
        mdf = pd.DataFrame(metrics).T[["rmse", "mape", "smape"]]
        st.dataframe(
            mdf.style.format({"rmse": "{:.0f}", "mape": "{:.2f}%", "smape": "{:.2f}%"}),
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Backtest skipped: {e}")

# ---- Fit & forecast ----
forecasts, notes = fit_and_forecast(ts, horizon=horizon, models=models)

if notes:
    st.info(" | ".join(notes))

if not forecasts:  # empty dict
    st.error("All models failed to run on this selection. Try a longer history or different filters.")
    st.stop()


# ---- Plot ----
fig = plot_actual_forecasts(ts, forecasts)
st.pyplot(fig)

with st.expander("XGBoost feature importance", expanded=False):
    if "XGBoost" in forecasts:
        try:
            from src.models import xgb_importance
            imp = xgb_importance(ts)
            if imp is None:
                st.info("Not enough history to compute importances.")
            else:
                st.bar_chart(imp)
        except Exception as e:
            st.info(f"Skipped: {e}")
    else:
        st.caption("Run XGBoost to see importances.")


# ---- Table + download ----
out = pd.DataFrame({"actual": ts})
future_idx = pd.period_range(ts.index[-1] + 1, periods=horizon, freq="M")

future_tbl = (
    out.loc[out.index.isin(future_idx)]
       .reset_index()
       .rename(columns={"index": "month"})
)

st.markdown("### Forecast table")
st.dataframe(
    future_tbl.style.format({c: "{:,.0f}" for c in future_tbl.columns if c != "month"}),
    use_container_width=True
)

st.download_button(
    "Download forecasts CSV",
    data=future_tbl.to_csv(index=False),
    file_name="forecasts_future.csv",
    mime="text/csv",
)

with st.expander("History (actuals)"):
    hist_tbl = (
        out.loc[out.index <= ts.index[-1]]
           .reset_index()
           .rename(columns={"index": "month"})
    )
    st.dataframe(
        hist_tbl.style.format({c: "{:,.0f}" for c in hist_tbl.columns if c != "month"}),
        use_container_width=True
    )
