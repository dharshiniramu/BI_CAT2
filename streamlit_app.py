from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Amazon Sales and Demand Forecasting",
    page_icon="📊",
    layout="wide",
)

from agent_ui import render_agent
from dashboards.sales import render_sales_dashboard
from data import (
    FORECAST_FILE,
    PREDICTIONS_FILE,
    SALES_FILE,
    load_predictions,
    load_sales_data,
)
from ml.dashboard import render_ml_dashboard


def require_files() -> bool:
    missing = [
        path.name
        for path in [SALES_FILE, PREDICTIONS_FILE, FORECAST_FILE]
        if not path.exists()
    ]
    if missing:
        st.error("Missing required file(s): " + ", ".join(missing))
        st.info("Run forecast.py first so the ML CSV outputs are available.")
        return False
    return True


def main() -> None:
    st.title("Amazon sales and demand forecasting")
    st.caption(
        "Interactive Python dashboard using the original sales data and "
        "the existing ML forecast outputs."
    )

    if not require_files():
        return

    sales = load_sales_data()
    predictions, forecast = load_predictions()

    with st.sidebar:
        st.header("Filters")
        categories = sorted(sales["Category"].dropna().unique())
        selected_categories = st.multiselect(
            "Sales categories",
            categories,
            default=categories,
        )
        statuses = sorted(sales["OrderStatus"].dropna().unique())
        selected_statuses = st.multiselect(
            "Order statuses",
            statuses,
            default=statuses,
        )

    filtered_sales = sales[
        sales["Category"].isin(selected_categories)
        & sales["OrderStatus"].isin(selected_statuses)
    ]

    active_tab = st.segmented_control(
        "Dashboard",
        options=["Sales overview", "ML demand forecast"],
        key="dashboard_tab",
        default="Sales overview",
        label_visibility="collapsed",
        width="stretch",
    )
    if active_tab is None:
        active_tab = "Sales overview"

    if active_tab == "Sales overview":
        render_sales_dashboard(filtered_sales)
    else:
        render_ml_dashboard(predictions, forecast)

    filtered_data = {
        "sales": filtered_sales,
        "predictions": predictions,
        "forecast": forecast,
    }
    render_agent(filtered_data, active_tab)


if __name__ == "__main__":
    main()
