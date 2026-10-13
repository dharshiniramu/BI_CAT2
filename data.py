from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SALES_FILE = BASE_DIR / "Amazon.xlsx"
PREDICTIONS_FILE = BASE_DIR / "demand_predictions.csv"
FORECAST_FILE = BASE_DIR / "product_demand_forecast.csv"


@st.cache_data
def load_sales_data() -> pd.DataFrame:
    data = pd.read_excel(SALES_FILE)
    data["OrderDate"] = pd.to_datetime(data["OrderDate"], errors="coerce")
    data = data.dropna(subset=["OrderDate"]).copy()
    data["OrderMonth"] = data["OrderDate"].dt.to_period("M").astype(str)
    data["EffectiveQuantity"] = data["Quantity"].where(
        ~data["OrderStatus"].str.lower().isin(["returned", "cancelled"]),
        0,
    )
    return data


@st.cache_data
def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")
    predictions["MonthLabel"] = predictions["Date"].dt.strftime("%Y-%m")

    forecast = pd.read_csv(FORECAST_FILE)
    forecast["Date"] = pd.to_datetime(forecast["Date"], errors="coerce")
    return predictions, forecast
