import pandas as pd
import plotly.express as px
import streamlit as st


def render_ml_dashboard(
    predictions: pd.DataFrame,
    forecast: pd.DataFrame,
) -> None:
    st.header("Machine-learning demand forecast")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted monthly records", f"{len(predictions):,}")
    col2.metric(
        "Average predicted demand",
        f"{predictions['PredictedDemand'].mean():,.1f} units",
    )
    col3.metric("Products in latest forecast", f"{len(forecast):,}")

    classification_counts = (
        forecast["DemandClassification"]
        .value_counts()
        .rename_axis("DemandClassification")
        .reset_index(name="Products")
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                classification_counts,
                x="DemandClassification",
                y="Products",
                color="DemandClassification",
                color_discrete_map={
                    "Low": "#4C78A8",
                    "Medium": "#F2CF5B",
                    "High": "#E45756",
                },
                title="Latest product demand classification",
            ),
            use_container_width=True,
        )
    with right:
        monthly_prediction = (
            predictions.groupby("MonthLabel", as_index=False)
            .agg(
                ActualDemand=("ActualDemand", "sum"),
                PredictedDemand=("PredictedDemand", "sum"),
            )
        )
        st.plotly_chart(
            px.line(
                monthly_prediction,
                x="MonthLabel",
                y=["ActualDemand", "PredictedDemand"],
                markers=True,
                title="Actual versus predicted monthly demand",
            ),
            use_container_width=True,
        )

    comparison = predictions[[
        "ProductName",
        "MonthLabel",
        "ActualDemand",
        "PredictedDemand",
        "DemandClassification",
    ]].copy()
    st.plotly_chart(
        px.scatter(
            comparison,
            x="ActualDemand",
            y="PredictedDemand",
            color="DemandClassification",
            hover_data=["ProductName", "MonthLabel"],
            color_discrete_map={
                "Low": "#4C78A8",
                "Medium": "#F2CF5B",
                "High": "#E45756",
            },
            title="Actual versus predicted demand by product-month",
        ),
        use_container_width=True,
    )

    st.subheader("Latest product forecast")
    display_forecast = forecast.sort_values(
        "PredictedDemand", ascending=False
    ).copy()
    st.dataframe(display_forecast, use_container_width=True, hide_index=True)
