import pandas as pd
import pytest

import agent_core


@pytest.fixture
def filtered_data():
    sales = pd.DataFrame(
        {
            "ProductID": [1, 2],
            "CustomerID": [10, 11],
            "TotalAmount": [120.0, 80.0],
            "OrderMonth": ["2026-01", "2026-01"],
            "EffectiveQuantity": [2, 1],
            "OrderID": [100, 101],
            "Category": ["Electronics", "Books"],
            "OrderStatus": ["Shipped", "Returned"],
            "ProductName": ["Headphones", "Notebook"],
        }
    )
    predictions = pd.DataFrame(
        {
            "ProductName": ["Headphones", "Notebook"],
            "MonthLabel": ["2026-01", "2026-01"],
            "ActualDemand": [8, 4],
            "PredictedDemand": [10, 3],
            "DemandClassification": ["High", "Low"],
        }
    )
    forecast = pd.DataFrame(
        {
            "ProductName": ["Headphones", "Notebook"],
            "Date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "PredictedDemand": [10, 3],
            "DemandClassification": ["High", "Low"],
            "SalesVelocity": [2.0, 1.0],
            "PreviousQuantity": [8, 4],
            "RollingAverage": [7.0, 3.0],
        }
    )
    return {
        "sales": sales,
        "predictions": predictions,
        "forecast": forecast,
    }


@pytest.fixture
def mock_answer(monkeypatch):
    monkeypatch.setattr(
        agent_core,
        "_generate_answer",
        lambda question, mode, context, history: f"{mode} answer",
    )


@pytest.mark.parametrize(
    ("mode", "question", "expected"),
    [
        ("sales", "Which category has the highest sales?", "sales"),
        ("forecast", "Explain the monthly demand chart", "forecast"),
        ("demand", "Which products have the highest predicted demand?", "demand"),
    ],
)
def test_in_scope_answers_use_the_matching_mode(
    mode, question, expected, filtered_data, mock_answer
):
    result = agent_core.answer(question, mode, filtered_data, [])
    assert result == {"text": f"{expected} answer", "redirect_to": None}


@pytest.mark.parametrize(
    ("mode", "question", "target"),
    [
        ("sales", "Explain the forecast chart", "forecast"),
        ("sales", "Which products have the highest predicted demand?", "demand"),
        ("forecast", "Which category has the highest sales?", "sales"),
        ("forecast", "Which products have the highest predicted demand?", "demand"),
        ("demand", "Explain the order-status chart", "sales"),
        ("demand", "Explain the monthly demand chart", "forecast"),
    ],
)
def test_questions_about_other_modes_redirect(mode, question, target, filtered_data):
    result = agent_core.answer(question, mode, filtered_data, [])
    assert result["redirect_to"] == target
    assert target.capitalize() in result["text"]


@pytest.mark.parametrize(
    ("mode", "question", "expected_mode", "redirect"),
    [
        ("sales", "Explain the monthly sales chart", "sales", None),
        ("sales", "Why is predicted demand high for product X?", "sales", "forecast"),
        ("sales", "Which products need restocking?", "sales", "demand"),
        ("forecast", "Explain the monthly sales chart", "forecast", "sales"),
        ("forecast", "Why is predicted demand high for product X?", "forecast", None),
        ("forecast", "Which products need restocking?", "forecast", "demand"),
        ("demand", "Explain the monthly sales chart", "demand", "sales"),
        ("demand", "Why is predicted demand high for product X?", "demand", "forecast"),
        ("demand", "Which products need restocking?", "demand", None),
    ],
)
def test_scope_examples_for_all_modes(
    mode, question, expected_mode, redirect, filtered_data, monkeypatch
):
    monkeypatch.setattr(
        agent_core,
        "_generate_answer",
        lambda question, answer_mode, context, history: f"{answer_mode} answer",
    )
    result = agent_core.answer(question, mode, filtered_data, [])
    if redirect is not None:
        assert result["redirect_to"] == redirect
    else:
        assert result == {"text": f"{expected_mode} answer", "redirect_to": None}


def test_empty_data_returns_filter_guidance(mock_answer):
    data = {
        "sales": pd.DataFrame(
            columns=[
                "ProductID",
                "CustomerID",
                "TotalAmount",
                "OrderMonth",
                "EffectiveQuantity",
                "OrderID",
                "Category",
                "OrderStatus",
                "ProductName",
            ]
        )
    }
    result = agent_core.answer("Explain this page", "sales", data, [])
    assert result["redirect_to"] is None
    assert "no data" in result["text"].lower()
    assert "widen" in result["text"].lower()


@pytest.mark.parametrize("mode", ["sales", "forecast"])
def test_empty_page_data_returns_filter_guidance(mode, filtered_data, monkeypatch):
    monkeypatch.setattr(
        agent_core,
        "_generate_answer",
        lambda question, answer_mode, context, history: "should not call LLM",
    )
    empty_data = {
        "sales": filtered_data["sales"].iloc[:0],
        "predictions": filtered_data["predictions"].iloc[:0],
        "forecast": filtered_data["forecast"].iloc[:0],
    }
    result = agent_core.answer("Explain this page", mode, empty_data, [])
    assert result["redirect_to"] is None
    assert "no data" in result["text"].lower()
    assert "widen" in result["text"].lower()


@pytest.mark.parametrize(
    ("mode", "expected_phrases"),
    [
        (
            "sales",
            ["200", "$200", "2026-01", "Electronics", "Shipped", "Headphones"],
        ),
        (
            "forecast",
            ["2 monthly demand estimates", "High", "2026-01", "Headphones"],
        ),
    ],
)
def test_page_summary_fallback_is_plain_data_backed_and_concise(
    mode, expected_phrases, filtered_data, monkeypatch
):
    monkeypatch.setattr(
        agent_core,
        "_generate_answer",
        lambda question, answer_mode, context, history: "",
    )
    result = agent_core.answer(
        "Explain this page", mode, filtered_data, []
    )
    assert result["redirect_to"] is None
    assert len(result["text"].split()) < 120
    assert "**What it means for you**" in result["text"]
    for phrase in expected_phrases:
        assert phrase.lower() in result["text"].lower()


def test_llm_key_lookup_failure_uses_data_backed_fallback(filtered_data, monkeypatch):
    monkeypatch.setattr(
        agent_core,
        "get_groq_api_key",
        lambda: (_ for _ in ()).throw(RuntimeError("secrets unavailable")),
    )
    result = agent_core.answer(
        "Summarize", "sales", filtered_data, []
    )
    assert "2 orders" in result["text"]


def test_broken_chart_data_returns_friendly_message():
    result = agent_core.answer(
        "Explain the monthly sales chart",
        "sales",
        {"sales": pd.DataFrame({"ProductID": [1]})},
        [],
    )
    assert "couldn't read" in result["text"].lower()


def test_off_topic_question_is_declined(filtered_data, monkeypatch):
    monkeypatch.setattr(
        agent_core, "_classify_ambiguous", lambda question, mode: "off_topic"
    )
    result = agent_core.answer("What is the weather tomorrow?", "sales", filtered_data, [])
    assert result["redirect_to"] is None
    assert "sales" in result["text"].lower()
    assert "forecast" in result["text"].lower()


def test_ambiguous_request_uses_classifier(filtered_data, monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent_core,
        "_classify_ambiguous",
        lambda question, mode: calls.append((question, mode)) or "sales",
    )
    monkeypatch.setattr(
        agent_core,
        "_generate_answer",
        lambda question, mode, context, history: "sales answer",
    )
    result = agent_core.answer("Can you help me understand this?", "sales", filtered_data, [])
    assert result == {"text": "sales answer", "redirect_to": None}
    assert calls == [("Can you help me understand this?", "sales")]
