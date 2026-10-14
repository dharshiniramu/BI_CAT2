"""Routing and answer generation for the floating dashboard assistant."""

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ai.demand_assistant import (
    build_demand_context,
    get_groq_api_key,
)
from chart_knowledge import CHART_REGISTRY, build_tab_context


MODE_LABELS = {
    "sales": "Sales",
    "forecast": "Forecast",
    "demand": "Demand",
}

SUGGESTIONS = {
    "sales": [
        "Explain this page in simple words",
        "Which category has the highest sales?",
        "Which country has the highest sales?",
    ],
    "forecast": [
        "Explain this page in simple words",
        "What does predicted versus actual demand mean?",
        "Which category has the highest predicted demand?",
    ],
    "demand": [
        "Which products have the highest predicted demand?",
        "Compare the demand for two products",
        "What should we do about products with high demand?",
    ],
}

OFF_TOPIC_REPLY = (
    "I can help explain the Sales and Forecast charts or answer product-demand questions."
)

_SALES_TERMS = re.compile(
    r"\b(sales?|revenue|orders?|customers?|categories|category|"
    r"order status|gross sales|units sold|top products by sales|"
    r"countries?|geography|payment method)\b",
    re.IGNORECASE,
)
_FORECAST_CHART_TERMS = re.compile(
    r"\b(chart|graph|plot|dashboard|page|bar|pie|scatter|table|"
    r"classification|classifications|actual(?:\s+versus|\s+vs\.?|\s+and)\s+predicted|"
    r"predicted(?:\s+versus|\s+vs\.?|\s+and)\s+actual|forecast tab|ml tab|"
    r"monthly demand|demand level distribution|prediction error|"
    r"forecast error|demand by category|forecast drivers|rolling average|"
    r"recent sales)\b",
    re.IGNORECASE,
)
_DEMAND_TERMS = re.compile(
    r"\b(highest demand|highest predicted demand|high demand|"
    r"which products?|compare .{0,60}products?|compare .{0,60}demand|"
    r"inventory|stock|restock|restock(?:ing)?|reorder|"
    r"what should (?:i|we) do|"
    r"expected demand for|demand for .{1,60})\b",
    re.IGNORECASE,
)
_PREDICTED_DEMAND_REASON = re.compile(
    r"\bwhy\b.{0,60}\b(predicted|forecast)\b.{0,40}\bdemand\b",
    re.IGNORECASE,
)
_CATEGORY_FORECAST_QUESTION = re.compile(
    r"\b(?:category|categories)\b.{0,50}\b(?:predicted|forecast)\s+demand\b|"
    r"\b(?:predicted|forecast)\s+demand\b.{0,50}\b(?:category|categories)\b",
    re.IGNORECASE,
)
_FORECAST_LEVEL_COUNT = re.compile(
    r"\b(how many|count|number of)\b.{0,50}\b(high|medium|low)\s+demand\b",
    re.IGNORECASE,
)
_FOLLOW_UP_TERMS = re.compile(
    r"^\s*(why|how so|what about|and |also\b|tell me more|"
    r"explain more|which one|what does that mean)\b",
    re.IGNORECASE,
)
_PAGE_SUMMARY_TERMS = re.compile(
    r"\b(explain this page|summari[sz]e|overview of this page|"
    r"all charts|whole page)\b",
    re.IGNORECASE,
)


def suggested_questions(mode: str) -> list[str]:
    """Return three starter questions for a supported assistant mode."""
    if mode not in SUGGESTIONS:
        raise ValueError("mode must be 'sales', 'forecast', or 'demand'")
    return list(SUGGESTIONS[mode])


def _cheap_scope(question: str, mode: str) -> str | None:
    """Return a clear mode, 'off_topic', or None when classification is ambiguous."""
    normalized = question.strip()
    lowered = normalized.lower()

    if _PAGE_SUMMARY_TERMS.search(lowered):
        return mode

    if _PREDICTED_DEMAND_REASON.search(normalized):
        return "forecast"
    if _CATEGORY_FORECAST_QUESTION.search(normalized):
        return "forecast"
    if _FORECAST_LEVEL_COUNT.search(normalized):
        return "forecast"

    sales_match = bool(_SALES_TERMS.search(normalized))
    forecast_chart_match = bool(_FORECAST_CHART_TERMS.search(normalized))
    demand_match = bool(_DEMAND_TERMS.search(normalized))

    if forecast_chart_match and sales_match and demand_match:
        return None
    if forecast_chart_match and (demand_match or not sales_match):
        return "forecast"
    if sales_match and not demand_match:
        return "sales"
    if demand_match and not sales_match:
        return "demand"
    if sales_match and demand_match:
        return None

    if _FOLLOW_UP_TERMS.search(normalized):
        return mode
    if lowered in {"explain this", "what about it?", "tell me more"}:
        return mode
    return None


def _classify_ambiguous(question: str, mode: str) -> str | None:
    """Use the configured LLM only when keyword routing cannot decide scope."""
    try:
        api_key = get_groq_api_key()
    except Exception:
        return None
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=20,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's request as exactly one label: "
                        "sales, forecast, demand, or off_topic. sales means "
                        "questions about sales, orders, customers, categories, "
                        "or their charts. forecast means questions explaining "
                        "forecast charts, recorded-versus-expected demand, or "
                        "demand-level chart counts. demand means product demand "
                        "questions, comparisons, or planning advice. Use "
                        "off_topic for unrelated requests. Reply with only "
                        "the label."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current assistant mode: {mode}\n"
                        f"Question: {question}"
                    ),
                },
            ],
        )
    except Exception:
        return None

    content = response.choices[0].message.content
    if not isinstance(content, str):
        return None
    result = content.strip().lower()
    match = re.search(r"\b(sales|forecast|demand|off_topic)\b", result)
    return match.group(1) if match else None


def _history_text(history: Sequence[Mapping[str, Any]], mode: str) -> str:
    relevant_messages = []
    for item in history[-12:]:
        if item.get("mode") not in (None, mode):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            relevant_messages.append(f"{role}: {content}")
    return "\n".join(relevant_messages)


def _mode_frames(
    mode: str,
    filtered_data: Mapping[str, pd.DataFrame],
) -> list[pd.DataFrame]:
    if mode == "sales":
        return [filtered_data["sales"]] if "sales" in filtered_data else []
    if mode == "forecast":
        return [
            filtered_data[key]
            for key in ("predictions", "forecast")
            if key in filtered_data
        ]
    return [
        filtered_data[key]
        for key in ("predictions", "forecast")
        if key in filtered_data
    ]


def _empty_data(frames: list[pd.DataFrame]) -> bool:
    return not frames or all(frame.empty for frame in frames)


def _required_data_missing(
    mode: str,
    filtered_data: Mapping[str, pd.DataFrame],
) -> bool:
    if mode == "sales":
        required = {"sales"}
    elif mode in {"forecast", "demand"}:
        required = {"predictions", "forecast"}
    else:
        return True
    return not required.issubset(filtered_data)


def _generate_answer(
    question: str,
    mode: str,
    context: str,
    history: str,
) -> str:
    """Ask Groq to answer from supplied data; return empty on provider failure."""
    try:
        api_key = get_groq_api_key()
    except Exception:
        return ""
    if not api_key:
        return ""

    try:
        from groq import Groq, NotFoundError
    except ImportError:
        return (
            "The Groq Python package is not installed in the active "
            "environment. Install it with `pip install groq`."
        )

    system_prompt = (
        "You are a friendly assistant for people with no analytics background. "
        "Answer only from the supplied dashboard data and chart descriptions; "
        "never invent numbers or facts. Use short, plain sentences and explain "
        "terms such as predicted (an estimate) versus actual (recorded) and "
        "classification (a simple low, medium, or high demand label). Unless "
        "the user asks for detail, stay under about 120 words. Structure an "
        "answer as one sentence saying what the chart shows, then 'What it "
        "means for you' with 2-3 bullets using numbers from the supplied "
        "context, then optionally 'What to watch out for'. For a request to "
        "explain or summarize the page, cover every chart in the supplied "
        "context in its listed order. Recommendations must be clearly labeled "
        "as suggestions, not measured facts. If data is missing, say so and "
        "suggest widening the filters. Use the recent conversation only to "
        "understand follow-up questions; current supplied data is authoritative."
    )
    user_prompt = (
        f"Assistant mode: {mode}\n"
        f"Recent conversation from this mode only:\n"
        f"{history or '(no earlier messages)'}\n\n"
        f"Current chart and data context:\n{context}\n\n"
        f"Question: {question}"
    )
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=700,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except NotFoundError:
        return ""
    except Exception as error:
        if getattr(error, "status_code", None) == 413:
            return ""
        return ""
    content = response.choices[0].message.content
    return content or ""


def _sales_page_summary(sales: pd.DataFrame) -> str:
    """Summarize the sales charts with live values in a short, fixed format."""
    monthly = (
        sales.groupby("OrderMonth", as_index=False)
        .agg(
            Sales=("TotalAmount", "sum"),
            Units=("EffectiveQuantity", "sum"),
            Orders=("OrderID", "nunique"),
        )
        .dropna(subset=["OrderMonth"])
    )
    categories = (
        sales.groupby("Category", as_index=False)["TotalAmount"]
        .sum()
        .sort_values("TotalAmount", ascending=False)
    )
    statuses = sales["OrderStatus"].value_counts()
    products = (
        sales.groupby("ProductName", as_index=False)["TotalAmount"]
        .sum()
        .sort_values("TotalAmount", ascending=False)
    )

    summary = [
        "This page shows selected orders, monthly sales, leading categories, order statuses, top products, countries, and payment methods.",
        "",
        "**What it means for you**",
        (
            f"- The selection has {len(sales):,} orders, "
            f"{sales['ProductID'].nunique():,} products, "
            f"{sales['CustomerID'].nunique():,} customers, and "
            f"${sales['TotalAmount'].sum():,.0f} in sales."
        ),
    ]
    if not monthly.empty:
        peak = monthly.loc[monthly["Sales"].idxmax()]
        trough = monthly.loc[monthly["Sales"].idxmin()]
        summary.append(
            f"- Monthly sales peaked in {peak['OrderMonth']} "
            f"(${peak['Sales']:,.0f}) and were lowest in "
            f"{trough['OrderMonth']} (${trough['Sales']:,.0f}); "
            f"the chart covers {monthly['Units'].sum():,.0f} units."
        )
    if not categories.empty and not statuses.empty and not products.empty:
        summary.append(
            f"- {categories.iloc[0]['Category']} led category sales; "
            f"{statuses.index[0]} was the most common order status; "
            f"{products.iloc[0]['ProductName']} led product sales."
        )
    if (
        not categories.empty
        and {"Country", "PaymentMethod"}.issubset(sales.columns)
    ):
        summary.append(
            f"- {categories.iloc[0]['Category']} is the leading category; "
            f"{sales.groupby('Country')['TotalAmount'].sum().idxmax()} is the leading country, "
            f"and {sales['PaymentMethod'].value_counts().index[0]} is the most common payment method."
        )
    return "\n".join(summary)


def _forecast_page_summary(
    predictions: pd.DataFrame,
    forecast: pd.DataFrame,
) -> str:
    """Summarize every forecast chart using only values from its chart frames."""
    levels = forecast["DemandClassification"].value_counts()
    monthly = (
        predictions.groupby("MonthLabel", as_index=False)
        .agg(
            ActualDemand=("ActualDemand", "sum"),
            PredictedDemand=("PredictedDemand", "sum"),
        )
        .dropna(subset=["MonthLabel"])
    )
    comparisons = predictions.copy()
    comparisons["ActualDemand"] = pd.to_numeric(
        comparisons["ActualDemand"], errors="coerce"
    )
    comparisons["PredictedDemand"] = pd.to_numeric(
        comparisons["PredictedDemand"], errors="coerce"
    )
    comparisons = comparisons.dropna(
        subset=["ActualDemand", "PredictedDemand"]
    )
    predicted = pd.to_numeric(
        predictions["PredictedDemand"], errors="coerce"
    ).dropna()
    top_forecast = forecast.sort_values(
        "PredictedDemand", ascending=False
    )

    monthly_reading = (
        (
            "Expected demand peaks in "
            f"{monthly.loc[monthly['PredictedDemand'].idxmax(), 'MonthLabel']} "
            f"at {monthly['PredictedDemand'].max():,.0f} units."
        )
        if not monthly.empty
        else "Monthly demand figures are not available."
    )
    summary = [
        "This page shows expected demand, demand levels, monthly comparisons, product forecasts, category demand, forecast drivers, and prediction error.",
        "",
        "**What it means for you**",
        (
            f"- The page includes {len(predictions):,} monthly demand estimates; "
            f"average expected demand is {predicted.mean():,.1f} units; "
            f"the latest forecast covers {len(forecast):,} products."
            if not predicted.empty
            else (
                f"- There are {len(predictions):,} monthly product records "
                f"and {len(forecast):,} products in the latest forecast."
            )
        ),
        (
            f"- Demand levels: {int(levels.get('High', 0)):,} high, "
            f"{int(levels.get('Medium', 0)):,} medium, and "
            f"{int(levels.get('Low', 0)):,} low. 'High' means more units "
            f"are expected than for the other levels. {monthly_reading}"
        ),
    ]
    category_totals = (
        forecast.groupby("Category")["PredictedDemand"].sum()
        if {"Category", "PredictedDemand"}.issubset(forecast.columns)
        else pd.Series(dtype=float)
    )
    error = (
        pd.to_numeric(predictions["PredictedDemand"], errors="coerce")
        - pd.to_numeric(predictions["ActualDemand"], errors="coerce")
    ).abs().dropna()
    if not category_totals.empty and not error.empty:
        summary.append(
            f"- {category_totals.idxmax()} has the highest category forecast "
            f"at {category_totals.max():,.0f} units; average prediction error is "
            f"{error.mean():,.1f} units."
        )
    if not comparisons.empty and not top_forecast.empty:
        average_gap = (
            comparisons["PredictedDemand"] - comparisons["ActualDemand"]
        ).abs().mean()
        highest_product = top_forecast.iloc[0]
        summary.append(
            f"- Recorded demand totals {monthly['ActualDemand'].sum():,.0f} "
            f"units versus {monthly['PredictedDemand'].sum():,.0f} expected; "
            f"the average difference for a product in a month is {average_gap:,.1f} "
            f"units; {highest_product['ProductName']} has the "
            f"highest latest estimate at "
            f"{highest_product['PredictedDemand']:,.0f} units."
        )
    return "\n".join(summary)


def _local_answer(
    question: str,
    mode: str,
    filtered_data: Mapping[str, pd.DataFrame],
    context: str,
) -> str:
    """Provide concise, data-backed output when the LLM is unavailable."""
    if _PAGE_SUMMARY_TERMS.search(question.lower()):
        if mode == "sales":
            return _sales_page_summary(filtered_data["sales"])
        if mode == "forecast":
            return _forecast_page_summary(
                filtered_data["predictions"],
                filtered_data["forecast"],
            )
    if mode in {"sales", "forecast"}:
        chart_id = _requested_chart(question, mode)
        entry = next(
            (item for item in CHART_REGISTRY if item["id"] == chart_id),
            None,
        )
        if entry is not None:
            facts = entry["get_insight"](filtered_data[entry["data_key"]])
            return "\n".join(
                [
                    f"{entry['title']} shows {entry['what_it_shows']}",
                    "",
                    "**What it means for you**",
                    *(f"- {fact}" for fact in facts),
                ]
            )
        return (
            "The AI service is unavailable, but here are the chart descriptions "
            "and current figures for this page. "
            "Ask about a specific chart for a shorter explanation.\n\n"
            + context
        )
    return (
        "I can't reach the demand assistant right now, but I can only answer "
        "from the available forecast figures. Please try again later."
    )


def _requested_chart(question: str, mode: str) -> str:
    """Select the most likely chart for a concise local explanation."""
    lowered = question.lower()
    if mode == "sales":
        if "month" in lowered:
            return "monthly-sales"
        if "countr" in lowered or "geograph" in lowered:
            return "country-sales"
        if "payment" in lowered:
            return "payment-methods"
        if "categor" in lowered:
            return "top-categories"
        if "status" in lowered:
            return "order-status"
        if "product" in lowered:
            return "top-products"
        return "sales-kpis"

    if any(term in lowered for term in ("classification", "high demand", "low demand", "medium demand")):
        return "demand-classification"
    if "error" in lowered:
        return "prediction-error"
    if "categor" in lowered:
        return "category-forecast"
    if any(term in lowered for term in ("driver", "rolling average", "recent sales")):
        return "forecast-drivers"
    if ("actual" in lowered or "predicted" in lowered) and "month" in lowered:
        return "monthly-actual-predicted"
    if ("scatter" in lowered or "product-month" in lowered or "compare product" in lowered):
        return "product-month-comparison"
    if "latest" in lowered or "table" in lowered or "product" in lowered:
        return "latest-product-forecast"
    return "forecast-kpis"


def _safe_local_answer(
    question: str,
    mode: str,
    filtered_data: Mapping[str, pd.DataFrame],
    context: str,
) -> str:
    """Build local fallback text without exposing dataframe errors in the UI."""
    try:
        return _local_answer(question, mode, filtered_data, context)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return (
            "I couldn't calculate an answer from the current data. "
            "Please check the data and try again."
        )


def _redirect_message(target_mode: str) -> str:
    if target_mode == "sales":
        subject = "the Sales charts"
    elif target_mode == "forecast":
        subject = "the Forecast charts"
    else:
        subject = "product demand"
    return (
        f"That's about {subject}. Switch to the {MODE_LABELS[target_mode]} "
        "mode of this assistant and I'll help with it."
    )


def answer(
    question: str,
    mode: str,
    filtered_data: Mapping[str, pd.DataFrame],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, str | None]:
    """Answer an in-scope question or return a redirect using current data.

    Provider and data-shape failures return a friendly data-backed fallback
    rather than breaking the Streamlit rerun.
    """
    if mode not in MODE_LABELS:
        raise ValueError("mode must be 'sales', 'forecast', or 'demand'")

    scope = _cheap_scope(question, mode)
    if scope is None:
        scope = _classify_ambiguous(question, mode)
        if scope is None:
            return {
                "text": (
                    "I couldn't tell which dashboard area you mean. "
                    "I can explain Sales charts, Forecast charts, or product demand."
                ),
                "redirect_to": None,
            }
    if scope == "off_topic":
        return {"text": OFF_TOPIC_REPLY, "redirect_to": None}
    if scope not in MODE_LABELS:
        return {
            "text": (
                "I couldn't tell which dashboard area you mean. "
                "I can explain Sales charts, Forecast charts, or product demand."
            ),
            "redirect_to": None,
        }
    if scope != mode:
        return {
            "text": _redirect_message(scope),
            "redirect_to": scope,
        }

    if _required_data_missing(mode, filtered_data):
        return {
            "text": (
                "The data needed for this view is not available right now. "
                "Please try again after the dashboard data has loaded."
            ),
            "redirect_to": None,
        }

    frames = _mode_frames(mode, filtered_data)
    if _empty_data(frames):
        return {
            "text": (
                "There is no data to explain for this selection. "
                "Try widening the filters and ask again."
            ),
            "redirect_to": None,
        }

    if mode == "sales":
        try:
            context = build_tab_context("sales", {"sales": filtered_data["sales"]})
        except (KeyError, TypeError, ValueError, AttributeError):
            return {
                "text": (
                    "I couldn't read the current sales figures. Please check "
                    "the data and try again."
                ),
                "redirect_to": None,
            }
    elif mode == "forecast":
        try:
            context = build_tab_context(
                "forecast",
                {
                    "predictions": filtered_data["predictions"],
                    "forecast": filtered_data["forecast"],
                },
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return {
                "text": (
                    "I couldn't read the current forecast figures. Please "
                    "check the data and try again."
                ),
                "redirect_to": None,
            }
    else:
        predictions = filtered_data["predictions"]
        forecast = filtered_data["forecast"]
        try:
            context = build_demand_context(question, predictions, forecast)
        except (KeyError, TypeError, ValueError, AttributeError):
            return {
                "text": (
                    "I couldn't read the current product-demand figures. "
                    "Please check the forecast data and try again."
                ),
                "redirect_to": None,
            }

    if mode in {"sales", "forecast"} and _PAGE_SUMMARY_TERMS.search(
        question.lower()
    ):
        return {
            "text": _safe_local_answer(question, mode, filtered_data, context),
            "redirect_to": None,
        }

    text = _generate_answer(
        question,
        mode,
        context,
        _history_text(history, mode),
    )
    if not text:
        text = _safe_local_answer(question, mode, filtered_data, context)
    return {
        "text": text,
        "redirect_to": None,
    }
