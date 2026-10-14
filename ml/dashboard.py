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

    left, right = st.columns(2)
    with left:
        category_forecast = (
            forecast.groupby("Category", as_index=False)
            .agg(
                PredictedDemand=("PredictedDemand", "sum"),
                Products=("ProductID", "nunique"),
            )
            .sort_values("PredictedDemand", ascending=False)
        )
        st.plotly_chart(
            px.bar(
                category_forecast.sort_values("PredictedDemand"),
                x="PredictedDemand",
                y="Category",
                orientation="h",
                hover_data=["Products"],
                title="Predicted demand by category",
            ),
            use_container_width=True,
        )
    with right:
        driver_data = forecast.copy()
        st.plotly_chart(
            px.scatter(
                driver_data,
                x="RollingAverage",
                y="PredictedDemand",
                size="SalesVelocity",
                color="DemandClassification",
                hover_data=["ProductName", "Category"],
                color_discrete_map={
                    "Low": "#4C78A8",
                    "Medium": "#F2CF5B",
                    "High": "#E45756",
                },
                title="Forecast demand versus recent sales",
                labels={
                    "RollingAverage": "Recent rolling-average demand",
                    "PredictedDemand": "Predicted demand",
                },
            ),
            use_container_width=True,
        )

    prediction_error = predictions.copy()
    prediction_error["AbsoluteError"] = (
        pd.to_numeric(prediction_error["PredictedDemand"], errors="coerce")
        - pd.to_numeric(prediction_error["ActualDemand"], errors="coerce")
    ).abs()
    error_by_month = (
        prediction_error.groupby("MonthLabel", as_index=False)
        .agg(MeanAbsoluteError=("AbsoluteError", "mean"))
        .dropna(subset=["MeanAbsoluteError"])
    )
    st.plotly_chart(
        px.bar(
            error_by_month,
            x="MonthLabel",
            y="MeanAbsoluteError",
            title="Average prediction error by month",
            labels={"MeanAbsoluteError": "Average absolute error (units)"},
        ),
        use_container_width=True,
    )

    st.subheader("Latest product forecast")
    display_forecast = forecast.sort_values(
        "PredictedDemand", ascending=False
    ).copy()
    st.dataframe(display_forecast, use_container_width=True, hide_index=True)
