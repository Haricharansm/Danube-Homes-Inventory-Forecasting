import streamlit as st
import pandas as pd

from src.data_loader import load_excel, monthly_aggregate
from src.forecasting import build_series, fit_and_forecast

st.title("📦 Batch Forecasts (All Store × Category)")

uploaded = st.file_uploader("Upload the same Excel/CSV you use on the Forecasts page", type=["xlsx","xls","csv"])
if uploaded is None:
    st.info("Upload your file to continue.")
    st.stop()

df, meta = load_excel(uploaded)
date_col, val_col = meta["date"], meta["value"]
store_col, group_col = meta["store"], meta["group"]
if not (date_col and val_col and store_col and group_col):
    st.error("Need Date (or Year/Month), Sales Val, Store, and Category/Group columns.")
    st.stop()

h = st.slider("Horizon (months)", 1, 12, 3)
models = ["SeasonalNaive", "Drift", "Holt", "AutoARIMA", "XGBoost"]
run = st.button("Run batch forecast")

if run:
    results = []
    groups = df[[store_col, group_col]].dropna().drop_duplicates().astype(str).values.tolist()
    pb = st.progress(0, text="Forecasting...")
    for i, (store, cat) in enumerate(groups, start=1):
        sub = df[(df[store_col].astype(str)==store) & (df[group_col].astype(str)==cat)].copy()
        monthly = monthly_aggregate(sub, date_col, val_col)
        if monthly.empty or len(monthly) < 3:
            continue
        ts = build_series(monthly, val_col)
        fdict, notes = fit_and_forecast(ts, horizon=h, models=models)
        for m, fc in fdict.items():
            for month, yhat in fc.items():
                results.append({
                    "store": store,
                    "category": cat,
                    "month": str(month),
                    "model": m,
                    "forecast": float(yhat)
                })
        pb.progress(i/len(groups), text=f"Forecasting {store} / {cat} ({i}/{len(groups)})")
    pb.empty()

    if len(results)==0:
        st.warning("No forecasts produced (not enough data per group).")
    else:
        out = pd.DataFrame(results)
        st.dataframe(out.head(50), use_container_width=True)
        st.download_button(
            "Download CSV",
            data=out.to_csv(index=False),
            file_name="batch_forecasts.csv",
            mime="text/csv"
        )
