"""Floating Streamlit interface for the chart and demand assistant."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from agent_core import answer, suggested_questions


MODES = ("sales", "forecast", "demand")
MODE_LABELS = {
    "sales": "📊 Sales",
    "forecast": "🔮 Forecast",
    "demand": "📦 Demand",
}
MODE_NAMES = {
    "sales": "Sales",
    "forecast": "Forecast",
    "demand": "Demand",
}
AGENT_TOOLTIP = "Ask me anything about charts, analytics and demand!"


def _default_mode(active_tab: str) -> str:
    tab = active_tab.strip().lower()
    if tab in {"sales", "sales overview"}:
        return "sales"
    if tab in {"forecast", "ml", "ml demand forecast"}:
        return "forecast"
    return "demand"


def _empty_history() -> dict[str, list[dict[str, str]]]:
    return {mode: [] for mode in MODES}


def _initialize_state(active_tab: str) -> None:
    if not isinstance(st.session_state.get("agent_open"), bool):
        st.session_state["agent_open"] = False
    if not isinstance(st.session_state.get("agent_mode_user_selected"), bool):
        st.session_state["agent_mode_user_selected"] = False
    if (
        not st.session_state["agent_mode_user_selected"]
        or st.session_state.get("agent_mode") not in MODES
    ):
        st.session_state["agent_mode"] = _default_mode(active_tab)
    history = st.session_state.get("agent_history")
    if not isinstance(history, dict):
        history = {}
    st.session_state["agent_history"] = {
        mode: (
            history.get(mode)
            if isinstance(history.get(mode), list)
            else []
        )
        for mode in MODES
    }

    redirects = st.session_state.get("agent_redirects")
    if not isinstance(redirects, dict):
        redirects = {}
    st.session_state["agent_redirects"] = {
        mode: (
            redirects.get(mode)
            if isinstance(redirects.get(mode), list)
            else []
        )
        for mode in MODES
    }


def _open_agent() -> None:
    st.session_state["agent_open"] = True


def _minimize_agent() -> None:
    st.session_state["agent_open"] = False


def _close_agent(default_mode: str) -> None:
    st.session_state["agent_history"] = _empty_history()
    st.session_state["agent_redirects"] = _empty_history()
    st.session_state["agent_mode"] = default_mode
    st.session_state["agent_mode_selector"] = default_mode
    st.session_state["agent_mode_user_selected"] = False
    st.session_state["agent_open"] = False
    st.session_state.pop("agent_pending_question", None)


def _select_mode(mode: str) -> None:
    st.session_state["agent_mode"] = mode
    st.session_state["agent_mode_selector"] = mode
    st.session_state["agent_mode_user_selected"] = True


def _mode_changed() -> None:
    st.session_state["agent_mode"] = st.session_state["agent_mode_selector"]
    st.session_state["agent_mode_user_selected"] = True


def _queue_suggestion(question: str) -> None:
    st.session_state["agent_pending_question"] = question


def _styles(is_open: bool) -> str:
    expanded_styles = ""
    if is_open:
        expanded_styles = """
        .st-key-agent_panel {
            position: fixed;
            top: 0;
            right: 0;
            z-index: 1000000;
            width: 420px;
            max-width: 100vw;
            height: 100vh;
            height: 100dvh;
            box-sizing: border-box;
            padding: 16px 18px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            color: #f4f4f5;
            background: #17191d;
            border-left: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: -12px 0 36px rgba(0, 0, 0, 0.28);
        }
        .st-key-agent_panel > div,
        .st-key-agent_panel > [data-testid="stVerticalBlock"],
        .st-key-agent_panel > div > [data-testid="stVerticalBlock"],
        .st-key-agent_panel > div > div[data-testid="stVerticalBlock"] {
            height: 100%;
            display: flex;
            flex-direction: column;
            min-height: 0;
            gap: 0.65rem;
        }
        .st-key-agent_panel div:has(> .st-key-agent_messages),
        .st-key-agent_panel div:has(> div > .st-key-agent_messages),
        .st-key-agent_panel [data-testid="stElementContainer"]:has(.st-key-agent_messages) {
            flex: 1 1 auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .st-key-agent_messages {
            flex: 1 1 auto;
            min-height: 0;
            height: 100%;
            display: flex;
            flex-direction: column;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-right: 8px;
            overscroll-behavior: contain;
            scrollbar-width: thin;
            scrollbar-color: #626873 #25282e;
        }
        .st-key-agent_messages > div,
        .st-key-agent_messages [data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
            width: 100%;
            min-height: min-content;
            height: auto;
            gap: 0.65rem;
        }
        .st-key-agent_messages::-webkit-scrollbar {
            width: 6px;
        }
        .st-key-agent_messages::-webkit-scrollbar-track {
            background: #25282e;
            border-radius: 3px;
        }
        .st-key-agent_messages::-webkit-scrollbar-thumb {
            background: #626873;
            border-radius: 3px;
        }
        .st-key-agent_messages::-webkit-scrollbar-thumb:hover {
            background: #808793;
        }
        .st-key-agent_messages [data-testid="stMarkdownContainer"],
        .st-key-agent_messages [data-testid="stChatMessage"],
        .st-key-agent_messages p,
        .st-key-agent_messages li,
        .st-key-agent_messages span {
            word-break: break-word;
            overflow-wrap: break-word;
        }
        .st-key-agent_messages table {
            display: block;
            width: 100%;
            max-width: 100%;
            overflow-x: auto;
            border-collapse: collapse;
        }
        .st-key-agent_messages table,
        .st-key-agent_messages pre,
        .st-key-agent_messages code,
        .st-key-agent_messages [data-testid="stCode"] {
            max-width: 100%;
            overflow-x: auto;
            word-break: break-word;
        }
        .st-key-agent_panel div:has(> .st-key-agent_chat_input),
        .st-key-agent_chat_input {
            flex: 0 0 auto;
            box-sizing: border-box;
            padding: 10px 0 0;
            background: #17191d;
            border-top: 1px solid rgba(255, 255, 255, 0.12);
            margin-top: auto;
        }
        .st-key-agent_chat_input [data-testid="stForm"] {
            background: #17191d;
            border: none;
            padding: 0;
        }
        .st-key-agent_panel,
        .st-key-agent_panel [data-testid="stMarkdownContainer"] p,
        .st-key-agent_panel [data-testid="stMarkdownContainer"] li,
        .st-key-agent_panel [data-testid="stMarkdownContainer"] h1,
        .st-key-agent_panel [data-testid="stMarkdownContainer"] h2,
        .st-key-agent_panel [data-testid="stMarkdownContainer"] h3,
        .st-key-agent_panel [data-testid="stMarkdownContainer"] h4,
        .st-key-agent_panel [data-testid="stMarkdownContainer"] span {
            color: #f4f4f5;
        }
        .st-key-agent_panel button {
            color: #f4f4f5;
            background-color: transparent;
            border-color: rgba(255, 255, 255, 0.18);
        }
        .st-key-agent_panel button:hover {
            border-color: #e53935;
            color: #ffffff;
            background-color: rgba(255, 255, 255, 0.08);
        }
        .st-key-agent_header {
            flex: 0 0 auto;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        }
        .st-key-agent_header h3 {
            margin: 0;
            white-space: nowrap;
            color: #f4f4f5 !important;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .st-key-agent_header [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        .st-key-agent_panel [data-testid="stSegmentedControl"] {
            flex: 0 0 auto;
        }
        .st-key-agent_suggestions button {
            width: 100%;
            min-height: 54px;
            white-space: normal;
            text-align: left;
        }
        .st-key-agent_panel iframe[title="st.components.v1.html"] {
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            border: none !important;
        }
        .stAppViewContainer .main .block-container {
            padding-right: 440px;
        }
        @media (max-width: 700px) {
            .st-key-agent_panel {
                width: 100vw;
                padding-left: 14px;
                padding-right: 14px;
            }
            .stAppViewContainer .main .block-container {
                padding-right: 1rem;
            }
        }
        """

    return f"""
    <style>
        .st-key-agent_launcher {{
            position: fixed;
            right: 24px;
            bottom: 24px;
            z-index: 1000001;
            width: 60px;
            height: 60px;
        }}
        .st-key-agent_launcher button {{
            width: 60px;
            height: 60px;
            min-height: 60px;
            padding: 0;
            border: 0;
            border-radius: 50%;
            color: #ffffff;
            background: #e53935;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.32);
            font-size: 25px;
        }}
        .st-key-agent_launcher button:hover {{
            color: #ffffff;
            background: #c62828;
            border: 0;
        }}
        {expanded_styles}
    </style>
    """


def _history_for_mode(mode: str) -> list[dict[str, str]]:
    """Return well-formed chat turns from the selected mode only."""
    history = st.session_state["agent_history"][mode]
    return [
        item
        for item in history
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]


def _render_history(mode: str) -> None:
    """Render the selected mode's conversation or starter-question buttons."""
    history = _history_for_mode(mode)
    redirects = st.session_state["agent_redirects"][mode]
    if not history:
        if st.session_state.get("agent_pending_question"):
            return
        st.markdown("Hi! Ask me to explain the charts or help with demand.")
        with st.container(key="agent_suggestions"):
            for index, prompt in enumerate(suggested_questions(mode)):
                st.button(
                    prompt,
                    key=f"agent_suggestion_{mode}_{index}",
                    on_click=_queue_suggestion,
                    args=(prompt,),
                    use_container_width=True,
                )
        return

    assistant_index = 0
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                redirect = (
                    redirects[assistant_index]
                    if assistant_index < len(redirects)
                    else None
                )
                if redirect in MODES:
                    st.button(
                        f"Switch to {MODE_NAMES[redirect]}",
                        key=f"agent_redirect_{mode}_{assistant_index}",
                        on_click=_select_mode,
                        args=(redirect,),
                    )
                assistant_index += 1


def _answer_question(
    question: str,
    mode: str,
    filtered_data: Mapping[str, pd.DataFrame],
) -> None:
    """Append the user turn and the answer from the current filtered data."""
    history = _history_for_mode(mode)
    st.session_state["agent_history"][mode].append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the charts and demand data..."):
            result = answer(question, mode, filtered_data, history)
        st.markdown(result["text"])
        redirect = result.get("redirect_to")
        if redirect in MODES:
            st.button(
                f"Switch to {MODE_NAMES[redirect]}",
                key=f"agent_redirect_new_{mode}_{len(history)}",
                on_click=_select_mode,
                args=(redirect,),
            )

    st.session_state["agent_history"][mode].append(
        {"role": "assistant", "content": result["text"]}
    )
    st.session_state["agent_redirects"][mode].append(
        redirect if redirect in MODES else None
    )


def _scroll_messages_to_bottom() -> None:
    st.components.v1.html(
        """
        <script>
        setTimeout(() => {
            const messages = window.parent.document.querySelector(
                ".st-key-agent_messages"
            );
            if (messages) {
                messages.scrollTop = messages.scrollHeight;
            }
        }, 100);
        </script>
        """,
        height=0,
    )


def render_agent(
    filtered_data: Mapping[str, pd.DataFrame],
    active_tab: str,
) -> None:
    """Render the floating launcher or expanded assistant panel.

    The application calls this on every rerun so chat submissions and changed
    sidebar filters use the latest data without manually triggering reruns.
    """
    _initialize_state(active_tab)
    is_open = st.session_state["agent_open"]
    st.markdown(_styles(is_open), unsafe_allow_html=True)

    if not is_open:
        st.session_state["agent_panel_was_open"] = False
        with st.container(key="agent_launcher"):
            st.button(
                "💬",
                key="agent_open_button",
                help=AGENT_TOOLTIP,
                on_click=_open_agent,
            )
        return

    was_open = st.session_state.get("agent_panel_was_open", False)
    should_scroll = False
    with st.container(key="agent_panel"):
        with st.container(key="agent_header"):
            title, minimize_column, close_column = st.columns([7, 1, 1])
            title.markdown("### Chart & Demand Assistant")
            minimize_column.button(
                "—",
                key="agent_minimize_button",
                help="Minimize",
                on_click=_minimize_agent,
            )
            close_column.button(
                "✕",
                key="agent_close_button",
                help="Close and clear conversation",
                on_click=_close_agent,
                args=(_default_mode(active_tab),),
            )

        st.session_state["agent_mode_selector"] = st.session_state["agent_mode"]
        st.segmented_control(
            "Assistant mode",
            options=list(MODES),
            format_func=lambda mode: MODE_LABELS[mode],
            key="agent_mode_selector",
            on_change=_mode_changed,
            label_visibility="collapsed",
            width="stretch",
        )

        mode = st.session_state["agent_mode"]
        if not was_open and _history_for_mode(mode):
            should_scroll = True
        messages_container = st.container(key="agent_messages")
        with messages_container:
            _render_history(mode)

        with st.form(key="agent_chat_input", clear_on_submit=True):
            question = st.text_input(
                "Ask about charts, analytics, or demand",
                key="agent_question_text",
                label_visibility="collapsed",
                placeholder="Ask about charts, analytics, or demand",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        queued_question = st.session_state.pop("agent_pending_question", None)
        if isinstance(queued_question, str):
            question = queued_question
        if question and (submitted or isinstance(queued_question, str)):
            with messages_container:
                _answer_question(question, mode, filtered_data)
            should_scroll = True

    st.session_state["agent_panel_was_open"] = True
    if should_scroll:
        _scroll_messages_to_bottom()
