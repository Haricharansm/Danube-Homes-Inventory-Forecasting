# Danube Homes — Advanced Forecasting Prototype (Streamlit)

This app provides demand forecasting for the Furniture & Home Furnishing dataset.

## Features
- **Autodetects columns**: Date (or Year/Month[/Day]), Store, Category/Group, Item, Sales Qty/Val.
- **Model zoo** with backtesting:
  - Seasonal Naive (baseline)
  - Auto ARIMA (`pmdarima`)
  - XGBoost (lags + calendar features)
  - (Optional) Prophet
- **Metrics**: RMSE, MAPE, sMAPE via expanding-window backtest.
- **Exports**: Forecast CSV.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
