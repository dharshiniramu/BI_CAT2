import pandas as pd
import plotly.express as px
import streamlit as st


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

    left, right = st.columns(2)
    with left:
        country_sales = (
            sales.groupby("Country", as_index=False)
            .agg(Sales=("TotalAmount", "sum"), Orders=("OrderID", "nunique"))
            .sort_values("Sales", ascending=False)
            .head(10)
        )
        st.plotly_chart(
            px.bar(
                country_sales.sort_values("Sales"),
                x="Sales",
                y="Country",
                orientation="h",
                hover_data=["Orders"],
                title="Top countries by sales",
            ),
            use_container_width=True,
        )
    with right:
        payment_mix = (
            sales.groupby("PaymentMethod", as_index=False)
            .agg(Orders=("OrderID", "nunique"), Sales=("TotalAmount", "sum"))
            .sort_values("Orders", ascending=False)
        )
        st.plotly_chart(
            px.bar(
                payment_mix,
                x="PaymentMethod",
                y="Orders",
                color="PaymentMethod",
                hover_data=["Sales"],
                title="Orders by payment method",
            ),
            use_container_width=True,
        )
