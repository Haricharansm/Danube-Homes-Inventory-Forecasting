import streamlit as st

st.set_page_config(page_title="Danube Homes Forecasting", layout="wide")
st.title("🏠 Danube Homes — Advanced Forecasting")

st.write("Use the sidebar to open **Forecasts** (Pages).")
st.markdown("""
**What you can do:**
- Load your Excel data (Date or Year/Month format).
- Choose models: Seasonal Naive, Auto ARIMA, XGBoost, + optional Prophet.
- Run backtests, compare RMSE/MAPE/sMAPE, and download forecasts.
""")
