from streamlit.testing.v1 import AppTest

import pandas as pd


def _agent_app():
    from agent_ui import render_agent
    import streamlit as st

    active_tab = st.session_state.get("test_active_tab", "Sales overview")
    render_agent(st.session_state.get("test_filtered_data", {}), active_tab)


def test_agent_launcher_opens_panel_and_initializes_state():
    app = AppTest.from_function(_agent_app, default_timeout=30).run()

    assert not app.exception
    assert app.session_state["agent_open"] is False
    assert app.session_state["agent_mode"] == "sales"
    assert set(app.session_state["agent_history"]) == {
        "sales",
        "forecast",
        "demand",
    }

    app.button(key="agent_open_button").click().run()

    assert not app.exception
    assert app.session_state["agent_open"] is True


def test_close_clears_all_modes_and_resets_active_mode():
    app = AppTest.from_function(_agent_app, default_timeout=30)
    app.session_state["agent_open"] = True
    app.session_state["agent_mode"] = "forecast"
    app.session_state["agent_mode_user_selected"] = True
    app.session_state["agent_history"] = {
        "sales": [{"role": "user", "content": "sales question"}],
        "forecast": [{"role": "user", "content": "forecast question"}],
        "demand": [{"role": "user", "content": "demand question"}],
    }
    app.run()

    app.button(key="agent_close_button").click().run()

    assert not app.exception
    assert app.session_state["agent_open"] is False
    assert app.session_state["agent_mode"] == "sales"
    assert app.session_state["agent_history"] == {
        "sales": [],
        "forecast": [],
        "demand": [],
    }


def test_first_open_uses_dashboard_and_manual_mode_survives_tab_switch():
    app = AppTest.from_function(_agent_app, default_timeout=30)
    app.session_state["test_active_tab"] = "ML demand forecast"
    app.session_state["agent_history"] = {
        "sales": [{"role": "user", "content": "sales history"}],
        "forecast": [{"role": "user", "content": "forecast history"}],
        "demand": [{"role": "user", "content": "demand history"}],
    }
    app.run()
    assert app.session_state["agent_mode"] == "forecast"

    app.button(key="agent_open_button").click().run()
    app.segmented_control(key="agent_mode_selector").set_value("demand").run()
    assert app.session_state["agent_mode"] == "demand"
    assert app.session_state["agent_mode_user_selected"] is True

    app.session_state["test_active_tab"] = "Sales overview"
    app.run()
    assert app.session_state["agent_mode"] == "demand"
    assert all(app.session_state["agent_history"][mode] for mode in (
        "sales",
        "forecast",
        "demand",
    ))

    app.button(key="agent_minimize_button").click().run()
    app.button(key="agent_open_button").click().run()
    assert app.session_state["agent_mode"] == "demand"
    assert all(app.session_state["agent_history"][mode] for mode in (
        "sales",
        "forecast",
        "demand",
    ))


def test_filter_changes_preserve_mode_histories():
    app = AppTest.from_function(_agent_app, default_timeout=30)
    app.session_state["agent_open"] = True
    app.session_state["agent_mode"] = "sales"
    app.session_state["agent_mode_user_selected"] = True
    app.session_state["agent_history"] = {
        "sales": [{"role": "assistant", "content": "Earlier sales answer"}],
        "forecast": [{"role": "assistant", "content": "Earlier forecast answer"}],
        "demand": [{"role": "assistant", "content": "Earlier demand answer"}],
    }
    app.run()

    app.session_state["test_filtered_data"] = {"sales": "updated selection"}
    app.run()

    assert app.session_state["agent_history"] == {
        "sales": [{"role": "assistant", "content": "Earlier sales answer"}],
        "forecast": [{"role": "assistant", "content": "Earlier forecast answer"}],
        "demand": [{"role": "assistant", "content": "Earlier demand answer"}],
    }


def test_next_answer_receives_updated_filtered_data_and_same_mode_history(monkeypatch):
    import agent_ui

    received = []

    def fake_answer(question, mode, filtered_data, history):
        received.append(
            (
                question,
                mode,
                filtered_data["sales"]["revision"].iloc[0],
                len(history),
            )
        )
        return {"text": "Answer based on the current selection.", "redirect_to": None}

    monkeypatch.setattr(agent_ui, "answer", fake_answer)
    app = AppTest.from_function(_agent_app, default_timeout=30)
    app.session_state["agent_open"] = True
    app.session_state["agent_mode"] = "sales"
    app.session_state["agent_mode_user_selected"] = True
    app.session_state["test_filtered_data"] = {
        "sales": pd.DataFrame({"revision": [1]})
    }
    app.run()

    app.text_input(key="agent_question_text").set_value(
        "Which category has the highest sales?"
    )
    app.button(key="FormSubmitter:agent_chat_input-Send").click().run()
    assert received == [
        ("Which category has the highest sales?", "sales", 1, 0)
    ]

    app.session_state["test_filtered_data"] = {
        "sales": pd.DataFrame({"revision": [2]})
    }
    app.run()
    app.text_input(key="agent_question_text").set_value("Why?")
    app.button(key="FormSubmitter:agent_chat_input-Send").click().run()

    assert received[-1] == ("Why?", "sales", 2, 2)


def test_long_conversation_scrolls_after_messages_and_once_after_reopen(monkeypatch):
    import agent_ui

    scrolls = []
    long_answer = "Long response with readable wrapped content. " * 250
    monkeypatch.setattr(agent_ui, "answer", lambda *args: {
        "text": long_answer,
        "redirect_to": None,
    })
    monkeypatch.setattr(
        agent_ui, "_scroll_messages_to_bottom", lambda: scrolls.append(True)
    )
    app = AppTest.from_function(_agent_app, default_timeout=30)
    app.session_state["agent_open"] = True
    app.session_state["agent_mode"] = "sales"
    app.session_state["agent_mode_user_selected"] = True
    app.run()

    assert scrolls == []
    for question in ["Explain this page", "Why?", "Why?", "Why?", "Why?", "Why?"]:
        app.text_input(key="agent_question_text").set_value(question)
        app.button(key="FormSubmitter:agent_chat_input-Send").click().run()

    assert scrolls == [True] * 6
    assert len(app.session_state["agent_history"]["sales"]) == 12
    assert app.session_state["agent_history"]["sales"][-1]["content"] == long_answer

    app.segmented_control(key="agent_mode_selector").set_value("forecast").run()
    assert len(scrolls) == 6
    app.segmented_control(key="agent_mode_selector").set_value("sales").run()
    app.button(key="agent_minimize_button").click().run()
    app.button(key="agent_open_button").click().run()
    assert len(scrolls) == 7
