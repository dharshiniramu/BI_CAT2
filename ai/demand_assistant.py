import os
import re

import pandas as pd
import streamlit as st


def get_groq_api_key() -> str | None:
    """Read the Groq key from the environment or Streamlit secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def build_demand_context(
    question: str,
    predictions: pd.DataFrame,
    forecast: pd.DataFrame,
) -> str:
    question_lower = question.lower()
    matching_products = [
        product
        for product in forecast["ProductName"].dropna().unique()
        if product.lower() in question_lower
    ]

    if matching_products:
        relevant_forecast = forecast[
            forecast["ProductName"].isin(matching_products)
        ]
        relevant_predictions = predictions[
            predictions["ProductName"].isin(matching_products)
        ]
    else:
        relevant_forecast = forecast
        relevant_predictions = predictions

    requested_month = None
    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+(\d{4})\b",
        question_lower,
    )
    if month_match:
        requested_month = pd.to_datetime(
            f"{month_match.group(1)} {month_match.group(2)}",
            format="%B %Y",
        ).strftime("%Y-%m")
        relevant_predictions = relevant_predictions[
            relevant_predictions["MonthLabel"] == requested_month
        ]
    elif "next month" in question_lower:
        latest_month = predictions["MonthLabel"].max()
        relevant_predictions = relevant_predictions[
            relevant_predictions["MonthLabel"] == latest_month
        ]

    requested_classification = next(
        (
            classification
            for classification in ["high", "medium", "low"]
            if classification in question_lower
        ),
        None,
    )
    if requested_classification:
        relevant_predictions = relevant_predictions[
            relevant_predictions["DemandClassification"].str.lower()
            == requested_classification
        ]
        relevant_forecast = relevant_forecast[
            relevant_forecast["DemandClassification"].str.lower()
            == requested_classification
        ]

    latest_columns = [
        "ProductName",
        "Date",
        "PredictedDemand",
        "DemandClassification",
        "SalesVelocity",
        "PreviousQuantity",
        "RollingAverage",
    ]
    latest_text = relevant_forecast[latest_columns].sort_values(
        "PredictedDemand", ascending=False
    ).head(20).to_csv(index=False)

    monthly_product_columns = [
        "ProductName",
        "MonthLabel",
        "ActualDemand",
        "PredictedDemand",
        "DemandClassification",
    ]
    monthly_product_data = relevant_predictions[monthly_product_columns].sort_values(
        "PredictedDemand", ascending=False
    ).head(60)
    monthly_product_text = monthly_product_data.to_csv(index=False)

    monthly_summary = (
        relevant_predictions.groupby("MonthLabel", as_index=False)
        .agg(
            ActualDemand=("ActualDemand", "sum"),
            PredictedDemand=("PredictedDemand", "sum"),
        )
        .to_csv(index=False)
    )

    classification_summary = (
        relevant_forecast["DemandClassification"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .to_string()
    )

    scope = (
        "The question mentions these products: "
        + ", ".join(matching_products)
        if matching_products
        else "No specific product was detected; use the complete forecast context."
    )
    month_scope = (
        f"The requested month is {requested_month}."
        if requested_month
        else (
            "For 'next month', use the latest available forecast month "
            f"({predictions['MonthLabel'].max()})."
            if "next month" in question_lower
            else "No specific month was requested."
        )
    )
    return (
        f"{scope}\n\n"
        f"{month_scope}\n\n"
        "Latest product forecast data:\n"
        f"{latest_text}\n"
        "Monthly product prediction data (filtered to the requested product "
        "or month when applicable; highest 60 rows shown):\n"
        f"{monthly_product_text}\n"
        "Monthly actual-versus-predicted totals:\n"
        f"{monthly_summary}\n"
        "Latest classification counts:\n"
        f"{classification_summary}"
    )
