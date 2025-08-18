import matplotlib.pyplot as plt
import pandas as pd

def plot_actual_forecasts(ts: pd.Series, forecasts: dict):
    fig, ax = plt.subplots()
    ax.plot(ts.index.to_timestamp(), ts.values, marker="o", label="Actual")
    for name, fc in forecasts.items():
        ax.plot(fc.index.to_timestamp(), fc.values, marker="o", label=name)
    ax.set_title("Actual vs Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()
    return fig
