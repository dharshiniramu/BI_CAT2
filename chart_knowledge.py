"""Plain-language descriptions and live insights for dashboard visualizations."""

from collections.abc import Mapping
from typing import Callable, TypedDict

import pandas as pd


class ChartKnowledgeEntry(TypedDict):
    id: str
    tab: str
    title: str
    what_it_shows: str
    how_to_read: str
    data_key: str
    get_insight: Callable[[pd.DataFrame], list[str]]


def _empty_insight(subject: str) -> list[str]:
    return [
        f"No {subject} are available for the current data.",
        "Change the filters or add data to see a comparison.",
    ]


def _number(value: float) -> str:
    return f"{value:,.0f}"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _sales_kpi_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("sales records")

    orders = len(filtered_df)
    products = filtered_df["ProductID"].nunique()
    customers = filtered_df["CustomerID"].nunique()
    gross_sales = filtered_df["TotalAmount"].sum()
    return [
        f"The current selection contains {orders:,} orders.",
        f"Those orders include {products:,} products from {customers:,} customers.",
        f"Gross sales in the current selection total {_money(gross_sales)}.",
    ]


def _monthly_sales_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("monthly sales figures")

    monthly = (
        filtered_df.groupby("OrderMonth", as_index=False)
        .agg(
            Sales=("TotalAmount", "sum"),
            Units=("EffectiveQuantity", "sum"),
            Orders=("OrderID", "nunique"),
        )
        .dropna(subset=["OrderMonth"])
    )
    if monthly.empty:
        return _empty_insight("monthly sales figures")

    high_sales = monthly.loc[monthly["Sales"].idxmax()]
    low_sales = monthly.loc[monthly["Sales"].idxmin()]
    return [
        f"Sales were highest in {high_sales['OrderMonth']} at {_money(high_sales['Sales'])}.",
        f"Sales were lowest in {low_sales['OrderMonth']} at {_money(low_sales['Sales'])}.",
        (
            f"Across the shown months, there were {_number(monthly['Units'].sum())} "
            f"units and {monthly['Orders'].sum():,} orders."
        ),
    ]


def _top_categories_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("category sales")

    categories = (
        filtered_df.groupby("Category", as_index=False)["TotalAmount"]
        .sum()
        .sort_values("TotalAmount", ascending=False)
        .head(10)
    )
    if categories.empty:
        return _empty_insight("category sales")

    first = categories.iloc[0]
    facts = [
        f"{first['Category']} is the top category at {_money(first['TotalAmount'])} in sales."
    ]
    if len(categories) > 1:
        second = categories.iloc[1]
        difference = first["TotalAmount"] - second["TotalAmount"]
        facts.append(
            f"It is ahead of {second['Category']} by {_money(difference)}."
        )
    facts.append(
        f"The {len(categories)} categories shown together account for "
        f"{_money(categories['TotalAmount'].sum())} in sales."
    )
    return facts[:3]


def _order_status_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("orders by status")

    status_counts = filtered_df["OrderStatus"].value_counts()
    if status_counts.empty:
        return _empty_insight("orders by status")

    total = int(status_counts.sum())
    leading_status = status_counts.index[0]
    leading_count = int(status_counts.iloc[0])
    share = leading_count / total * 100
    facts = [
        f"{leading_status} is the most common order status, with {leading_count:,} orders ({share:.1f}%)."
    ]
    if len(status_counts) > 1:
        next_status = status_counts.index[1]
        next_count = int(status_counts.iloc[1])
        facts.append(
            f"{next_status} follows with {next_count:,} orders."
        )
    facts.append(f"The chart includes {total:,} orders in total.")
    return facts[:3]


def _top_products_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("product sales")

    products = (
        filtered_df.groupby("ProductName", as_index=False)
        .agg(Sales=("TotalAmount", "sum"), Units=("EffectiveQuantity", "sum"))
        .sort_values("Sales", ascending=False)
        .head(15)
    )
    if products.empty:
        return _empty_insight("product sales")

    first = products.iloc[0]
    facts = [
        f"{first['ProductName']} leads the products shown with {_money(first['Sales'])} in sales."
    ]
    if len(products) > 1:
        second = products.iloc[1]
        facts.append(
            f"{second['ProductName']} is next at {_money(second['Sales'])}."
        )
    facts.append(
        f"The products shown account for {_number(products['Units'].sum())} units sold."
    )
    return facts[:3]


def _country_sales_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty or not {"Country", "TotalAmount"}.issubset(filtered_df.columns):
        return _empty_insight("country sales")

    countries = (
        filtered_df.groupby("Country", as_index=False)["TotalAmount"]
        .sum()
        .sort_values("TotalAmount", ascending=False)
        .head(10)
    )
    if countries.empty:
        return _empty_insight("country sales")
    first = countries.iloc[0]
    return [
        f"{first['Country']} leads the countries shown at {_money(first['TotalAmount'])}.",
        f"The top {len(countries)} countries together account for {_money(countries['TotalAmount'].sum())}.",
    ]


def _payment_method_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty or not {"PaymentMethod", "OrderID"}.issubset(filtered_df.columns):
        return _empty_insight("payment-method orders")

    methods = filtered_df["PaymentMethod"].value_counts()
    if methods.empty:
        return _empty_insight("payment-method orders")
    first = methods.index[0]
    share = methods.iloc[0] / methods.sum() * 100
    return [
        f"{first} is the most common payment method with {methods.iloc[0]:,} orders ({share:.1f}%).",
        f"The chart covers {methods.sum():,} orders across {len(methods):,} payment methods.",
    ]


def _forecast_kpi_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("forecast records")

    predicted_values = pd.to_numeric(
        filtered_df["PredictedDemand"], errors="coerce"
    ).dropna()
    facts = [
        f"The forecast contains {len(filtered_df):,} monthly product records."
    ]
    if not predicted_values.empty:
        facts.append(
            f"Average predicted demand is {predicted_values.mean():,.1f} units."
        )
    else:
        facts.append("Average predicted demand is not available for these records.")
    if "ProductName" in filtered_df:
        facts.append(
            f"The latest forecast covers {filtered_df['ProductName'].nunique():,} products."
        )
    return facts


def _classification_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("product demand levels")

    counts = filtered_df["DemandClassification"].value_counts()
    total_products = len(filtered_df)
    return [
        f"{int(counts.get('High', 0)):,} of {total_products:,} products are marked as high demand.",
        f"{int(counts.get('Medium', 0)):,} products are marked as medium demand.",
        f"{int(counts.get('Low', 0)):,} products are marked as low demand.",
    ]


def _monthly_demand_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("monthly demand figures")

    monthly = (
        filtered_df.groupby("MonthLabel", as_index=False)
        .agg(
            ActualDemand=("ActualDemand", "sum"),
            PredictedDemand=("PredictedDemand", "sum"),
        )
        .dropna(subset=["MonthLabel"])
    )
    if monthly.empty:
        return _empty_insight("monthly demand figures")

    highest = monthly.loc[monthly["PredictedDemand"].idxmax()]
    actual_total = monthly["ActualDemand"].sum()
    predicted_total = monthly["PredictedDemand"].sum()
    return [
        (
            f"Predicted demand is highest in {highest['MonthLabel']} at "
            f"{_number(highest['PredictedDemand'])} units."
        ),
        (
            f"Across the shown months, predicted demand totals "
            f"{_number(predicted_total)} units and actual demand totals "
            f"{_number(actual_total)} units."
        ),
    ]


def _product_month_comparison_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("product-month comparisons")

    comparisons = filtered_df.copy()
    comparisons["ActualDemand"] = pd.to_numeric(
        comparisons["ActualDemand"], errors="coerce"
    )
    comparisons["PredictedDemand"] = pd.to_numeric(
        comparisons["PredictedDemand"], errors="coerce"
    )
    comparisons = comparisons.dropna(
        subset=["ActualDemand", "PredictedDemand"]
    )
    if comparisons.empty:
        return _empty_insight("product-month comparisons")

    comparisons["Difference"] = (
        comparisons["PredictedDemand"] - comparisons["ActualDemand"]
    )
    over_predictions = int((comparisons["Difference"] > 0).sum())
    largest_gap = comparisons.loc[comparisons["Difference"].abs().idxmax()]
    direction = (
        "higher"
        if largest_gap["Difference"] > 0
        else "lower"
        if largest_gap["Difference"] < 0
        else "the same as"
    )
    facts = [
        (
            f"Predicted demand is higher than actual demand in {over_predictions} "
            f"of {len(comparisons):,} product-month comparisons."
        ),
    ]
    if largest_gap["Difference"] == 0:
        facts.append(
            f"{largest_gap['ProductName']} in {largest_gap['MonthLabel']} has matching actual and predicted demand."
        )
    else:
        facts.append(
            f"The largest gap is for {largest_gap['ProductName']} in "
            f"{largest_gap['MonthLabel']}: predicted demand is "
            f"{_number(abs(largest_gap['Difference']))} units {direction} than actual."
        )
    average_gap = comparisons["Difference"].abs().mean()
    facts.append(
        f"The average gap between predicted and actual demand is {average_gap:,.1f} units."
    )
    return facts[:3]


def _latest_forecast_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty:
        return _empty_insight("latest forecast rows")

    forecasts = filtered_df.sort_values("PredictedDemand", ascending=False)
    highest = forecasts.iloc[0]
    facts = [
        (
            f"{highest['ProductName']} has the highest predicted demand "
            f"at {_number(highest['PredictedDemand'])} units."
        ),
        f"The table contains {len(forecasts):,} product forecasts.",
    ]
    if len(forecasts) > 1:
        lowest = forecasts.iloc[-1]
        facts.append(
            f"{lowest['ProductName']} has the lowest predicted demand "
            f"at {_number(lowest['PredictedDemand'])} units."
        )
    return facts[:3]


def _category_forecast_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty or not {
        "Category",
        "PredictedDemand",
        "ProductID",
    }.issubset(filtered_df.columns):
        return _empty_insight("category forecasts")

    categories = (
        filtered_df.groupby("Category", as_index=False)
        .agg(PredictedDemand=("PredictedDemand", "sum"), Products=("ProductID", "nunique"))
        .sort_values("PredictedDemand", ascending=False)
    )
    if categories.empty:
        return _empty_insight("category forecasts")
    first = categories.iloc[0]
    return [
        f"{first['Category']} has the highest expected demand at {_number(first['PredictedDemand'])} units.",
        f"The chart compares {len(categories):,} categories across {int(categories['Products'].sum()):,} products.",
    ]


def _forecast_driver_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty or not {
        "RollingAverage",
        "PredictedDemand",
        "SalesVelocity",
    }.issubset(filtered_df.columns):
        return _empty_insight("forecast driver relationships")

    correlation = filtered_df[["RollingAverage", "PredictedDemand"]].corr().iloc[0, 1]
    if pd.isna(correlation):
        return ["There are not enough numeric values to compare recent and predicted demand."]
    direction = "positive" if correlation >= 0 else "negative"
    return [
        f"The relationship between recent rolling-average demand and predicted demand is {direction} (correlation {correlation:.2f}).",
        "Larger bubbles indicate higher sales velocity; colors identify low, medium, and high demand.",
    ]


def _prediction_error_insight(filtered_df: pd.DataFrame) -> list[str]:
    if filtered_df.empty or not {
        "ActualDemand",
        "PredictedDemand",
        "MonthLabel",
    }.issubset(filtered_df.columns):
        return _empty_insight("prediction errors")

    actual = pd.to_numeric(filtered_df["ActualDemand"], errors="coerce")
    predicted = pd.to_numeric(filtered_df["PredictedDemand"], errors="coerce")
    errors = (predicted - actual).abs()
    valid = errors.dropna()
    if valid.empty:
        return _empty_insight("prediction errors")
    highest = valid.idxmax()
    return [
        f"Average absolute prediction error is {valid.mean():,.1f} units across {len(valid):,} product-month records.",
        f"The largest absolute error is {valid.loc[highest]:,.1f} units in {filtered_df.loc[highest, 'MonthLabel']}.",
    ]


CHART_REGISTRY: list[ChartKnowledgeEntry] = [
    {
        "id": "sales-kpis",
        "tab": "sales",
        "title": "Sales overview key figures",
        "what_it_shows": "These figures summarize orders, products, customers, and sales for the selected records.",
        "how_to_read": "The four cards show order count, distinct products, distinct customers, and total sales; higher counts or sales mean more activity.",
        "data_key": "sales",
        "get_insight": _sales_kpi_insight,
    },
    {
        "id": "monthly-sales",
        "tab": "sales",
        "title": "Monthly sales, units, and orders",
        "what_it_shows": "This chart compares sales, units sold, and order counts across months.",
        "how_to_read": "Months run along the bottom and values run up the side; line colors identify sales, units, or orders, and a higher point means a larger value for that measure.",
        "data_key": "sales",
        "get_insight": _monthly_sales_insight,
    },
    {
        "id": "top-categories",
        "tab": "sales",
        "title": "Top categories by sales",
        "what_it_shows": "This chart ranks the ten categories with the highest sales.",
        "how_to_read": "Category names run up the side and sales run along the bottom; longer bars mean more sales.",
        "data_key": "sales",
        "get_insight": _top_categories_insight,
    },
    {
        "id": "order-status",
        "tab": "sales",
        "title": "Order-status distribution",
        "what_it_shows": "This chart shows how the selected orders are divided among their statuses.",
        "how_to_read": "Each color represents a status; a larger slice means more orders have that status.",
        "data_key": "sales",
        "get_insight": _order_status_insight,
    },
    {
        "id": "top-products",
        "tab": "sales",
        "title": "Top products by sales",
        "what_it_shows": "This chart ranks up to fifteen products by their sales and also shows units sold.",
        "how_to_read": "Product names run up the side and sales run along the bottom; longer bars mean more sales, and hovering shows units sold.",
        "data_key": "sales",
        "get_insight": _top_products_insight,
    },
    {
        "id": "country-sales",
        "tab": "sales",
        "title": "Top countries by sales",
        "what_it_shows": "This chart ranks the countries contributing the most sales.",
        "how_to_read": "Longer horizontal bars mean more sales; hovering also shows the number of orders.",
        "data_key": "sales",
        "get_insight": _country_sales_insight,
    },
    {
        "id": "payment-methods",
        "tab": "sales",
        "title": "Orders by payment method",
        "what_it_shows": "This chart compares order volume across payment methods.",
        "how_to_read": "Taller bars mean more orders; hovering shows the sales total for each method.",
        "data_key": "sales",
        "get_insight": _payment_method_insight,
    },
    {
        "id": "forecast-kpis",
        "tab": "forecast",
        "title": "Forecast key figures",
        "what_it_shows": "These figures summarize the forecast records, average expected demand, and products in the latest forecast.",
        "how_to_read": "The cards show monthly record count, average expected units, and product count; larger numbers mean more records, higher average demand, or more products.",
        "data_key": "predictions",
        "get_insight": _forecast_kpi_insight,
    },
    {
        "id": "demand-classification",
        "tab": "forecast",
        "title": "Latest product demand levels",
        "what_it_shows": "This chart counts products expected to have low, medium, or high demand.",
        "how_to_read": "Demand levels run along the bottom and product counts run up the side; blue means low, yellow means medium, and red means high demand, while taller bars mean more products.",
        "data_key": "forecast",
        "get_insight": _classification_insight,
    },
    {
        "id": "monthly-actual-predicted",
        "tab": "forecast",
        "title": "Actual and predicted monthly demand",
        "what_it_shows": "This chart compares recorded demand with expected demand for each month.",
        "how_to_read": "Months run along the bottom and units run up the side; separate lines show actual and predicted demand, and higher points mean more units.",
        "data_key": "predictions",
        "get_insight": _monthly_demand_insight,
    },
    {
        "id": "product-month-comparison",
        "tab": "forecast",
        "title": "Actual and predicted demand by product and month",
        "what_it_shows": "Each point compares actual demand with predicted demand for one product in one month.",
        "how_to_read": "Actual demand runs along the bottom and predicted demand runs up the side; points closer to the diagonal match more closely, colors show the demand level, and points higher or lower than the diagonal indicate predictions above or below actual demand.",
        "data_key": "predictions",
        "get_insight": _product_month_comparison_insight,
    },
    {
        "id": "latest-product-forecast",
        "tab": "forecast",
        "title": "Latest product forecast",
        "what_it_shows": "This table lists each product's latest expected demand and related forecast details.",
        "how_to_read": "Rows are ordered from highest to lowest predicted demand; larger demand values indicate more units are expected.",
        "data_key": "forecast",
        "get_insight": _latest_forecast_insight,
    },
    {
        "id": "category-forecast",
        "tab": "forecast",
        "title": "Predicted demand by category",
        "what_it_shows": "This chart totals expected demand for each product category.",
        "how_to_read": "Longer bars mean more units are expected in that category; hovering shows the product count.",
        "data_key": "forecast",
        "get_insight": _category_forecast_insight,
    },
    {
        "id": "forecast-drivers",
        "tab": "forecast",
        "title": "Forecast demand versus recent sales",
        "what_it_shows": "This chart compares predicted demand with each product's recent rolling-average demand.",
        "how_to_read": "Points farther right have higher recent demand, higher points have higher predictions, and larger bubbles have higher sales velocity.",
        "data_key": "forecast",
        "get_insight": _forecast_driver_insight,
    },
    {
        "id": "prediction-error",
        "tab": "forecast",
        "title": "Average prediction error by month",
        "what_it_shows": "This chart measures the average absolute gap between recorded and predicted demand each month.",
        "how_to_read": "Shorter bars mean predictions were closer to actual demand; taller bars mean larger average errors.",
        "data_key": "predictions",
        "get_insight": _prediction_error_insight,
    },
]


def build_tab_context(
    tab: str,
    filtered_data: Mapping[str, pd.DataFrame],
) -> str:
    """Build compact chart context from the same frames used to render a tab.

    Pass ``{"sales": filtered_sales}`` for the sales tab, or pass
    ``{"predictions": predictions, "forecast": forecast}`` for the forecast tab.
    """
    if tab not in {"sales", "forecast"}:
        raise ValueError("tab must be 'sales' or 'forecast'")

    entries = [entry for entry in CHART_REGISTRY if entry["tab"] == tab]
    required_keys = {entry["data_key"] for entry in entries}
    missing_keys = required_keys - filtered_data.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing data frame(s) for tab '{tab}': {missing}")

    sections = []
    for entry in entries:
        facts = entry["get_insight"](filtered_data[entry["data_key"]])
        sections.append(
            "\n".join(
                [
                    f"Chart: {entry['title']}",
                    f"What it shows: {entry['what_it_shows']}",
                    f"How to read it: {entry['how_to_read']}",
                    "Current insights: " + " ".join(facts),
                ]
            )
        )
    return "\n\n".join(sections)
