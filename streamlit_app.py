from pathlib import Path
import os
import re

from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
SALES_FILE = BASE_DIR / "Amazon.xlsx"
PREDICTIONS_FILE = BASE_DIR / "demand_predictions.csv"
FORECAST_FILE = BASE_DIR / "product_demand_forecast.csv"


st.set_page_config(
    page_title="Amazon Sales and Demand Forecasting",
    page_icon="📊",
    layout="wide",
)


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


def get_groq_api_key() -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets.get("GROQ_API_KEY")
    except (FileNotFoundError, KeyError):
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


def ask_demand_agent(
    question: str,
    predictions: pd.DataFrame,
    forecast: pd.DataFrame,
) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        return (
            "The demand assistant is not configured yet. Set the "
            "`GROQ_API_KEY` environment variable and restart Streamlit."
        )

    try:
        from groq import Groq, NotFoundError
    except ImportError:
        return (
            "The Groq Python package is not installed in the active "
            "environment. Install it with `pip install groq`."
        )

    context = build_demand_context(question, predictions, forecast)
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=700,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise demand analytics assistant. Answer "
                        "only from the supplied forecast context. If the context "
                        "does not contain the answer, say that it is not "
                        "available. You may provide practical inventory and "
                        "planning recommendations when the user asks what to do, "
                        "but label them as recommendations and base them on the "
                        "reported demand level. Do not invent data or claim that "
                        "a recommendation is a measured result. Distinguish actual demand "
                        "from predicted demand, and mention the relevant month "
                        "or product when applicable. For 'next month', use the "
                        "latest forecast month in the context and state that "
                        "month explicitly. Keep answers very concise: use at "
                        "most 5 bullets, list at most 10 products, and avoid "
                        "repeating the full dataset. For comparing products, "
                        "use a compact table with predicted demand, class, "
                        "and one-sentence conclusion."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Forecast context:\n{context}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
        )
    except NotFoundError:
        return (
            f"Groq cannot access model `{model}` for this API key. "
            "Update GROQ_MODEL in `.env` to `openai/gpt-oss-20b`, "
            "then restart Streamlit."
        )
    except Exception as error:
        if getattr(error, "status_code", None) == 413:
            return (
                "The question is valid, but the forecast context was too "
                "large for the current Groq free-tier token limit. Try "
                "mentioning a product, month, or classification such as "
                "`high demand`."
            )
        return (
            "The demand assistant could not complete the request. "
            f"Groq returned: {error}"
        )
    return response.choices[0].message.content


def render_demand_agent(
    predictions: pd.DataFrame,
    forecast: pd.DataFrame,
) -> None:
    st.header("Demand assistant")
    st.caption(
        "Ask about predicted demand, classifications, products, or "
        "actual-versus-predicted monthly demand."
    )
    st.caption(
        "Try: “Which products are expected to have high demand next month?”, "
        "“What should we do about high predicted demand?”, or "
        "“Compare Wireless Earbuds and Laptop Sleeve.”"
    )

    if not get_groq_api_key():
        st.info(
            "To enable the assistant, set `GROQ_API_KEY` before starting "
            "Streamlit. The key is read from the environment and is not "
            "stored in the app."
        )

    for message in st.session_state.get("demand_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Example: Which products have the highest predicted demand?"
    )
    if question:
        st.session_state.setdefault("demand_messages", []).append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Checking the ML forecast data..."):
                answer = ask_demand_agent(question, predictions, forecast)
            st.markdown(answer)
        st.session_state["demand_messages"].append(
            {"role": "assistant", "content": answer}
        )


def render_sales_dashboard(sales: pd.DataFrame) -> None:
    st.header("Original sales data")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Orders", f"{len(sales):,}")
    col2.metric("Products", f"{sales['ProductID'].nunique():,}")
    col3.metric("Customers", f"{sales['CustomerID'].nunique():,}")
    col4.metric("Gross sales", f"${sales['TotalAmount'].sum():,.0f}")

    monthly_sales = (
        sales.groupby("OrderMonth", as_index=False)
        .agg(
            Sales=("TotalAmount", "sum"),
            Units=("EffectiveQuantity", "sum"),
            Orders=("OrderID", "nunique"),
        )
    )
    monthly_sales_long = monthly_sales.melt(
        id_vars="OrderMonth",
        value_vars=["Sales", "Units", "Orders"],
        var_name="Metric",
        value_name="Value",
    )
    st.plotly_chart(
        px.line(
            monthly_sales_long,
            x="OrderMonth",
            y="Value",
            color="Metric",
            markers=True,
            title="Monthly sales, units, and orders",
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        category_sales = (
            sales.groupby("Category", as_index=False)["TotalAmount"]
            .sum()
            .sort_values("TotalAmount", ascending=False)
            .head(10)
        )
        st.plotly_chart(
            px.bar(
                category_sales,
                x="TotalAmount",
                y="Category",
                orientation="h",
                title="Top categories by sales",
            ),
            use_container_width=True,
        )
    with right:
        status_counts = sales["OrderStatus"].value_counts().rename_axis(
            "OrderStatus"
        ).reset_index(name="Orders")
        st.plotly_chart(
            px.pie(
                status_counts,
                names="OrderStatus",
                values="Orders",
                title="Order-status distribution",
            ),
            use_container_width=True,
        )

    product_sales = (
        sales.groupby("ProductName", as_index=False)
        .agg(Sales=("TotalAmount", "sum"), Units=("EffectiveQuantity", "sum"))
        .sort_values("Sales", ascending=False)
        .head(15)
    )
    st.plotly_chart(
        px.bar(
            product_sales.sort_values("Sales"),
            x="Sales",
            y="ProductName",
            orientation="h",
            hover_data=["Units"],
            title="Top products by sales",
        ),
        use_container_width=True,
    )


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

    tab_sales, tab_ml, tab_agent = st.tabs(
        ["Sales overview", "ML demand forecast", "Demand assistant"]
    )
    with tab_sales:
        render_sales_dashboard(filtered_sales)
    with tab_ml:
        render_ml_dashboard(predictions, forecast)
    with tab_agent:
        render_demand_agent(predictions, forecast)


if __name__ == "__main__":
    main()
