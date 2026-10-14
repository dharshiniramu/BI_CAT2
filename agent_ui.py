"""Floating Streamlit interface for the chart and demand assistant."""

from collections.abc import Mapping

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
        :root {
            --agent-w: clamp(380px, 30vw, 560px);
        }
        [data-testid="stAppViewContainer"] {
            width: calc(100vw - var(--agent-w)) !important;
            max-width: calc(100vw - var(--agent-w)) !important;
            overflow-x: hidden !important;
        }
        [data-testid="stMain"] {
            flex: 1 1 0 !important;
            min-width: 0 !important;
        }
        [data-testid="stHeader"],
        [data-testid="stBottom"] {
            width: calc(100vw - var(--agent-w)) !important;
            max-width: calc(100vw - var(--agent-w)) !important;
            left: 0 !important;
            right: auto !important;
        }
        .block-container {
            max-width: 100% !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        .st-key-agent_panel {
            position: fixed !important;
            top: 0 !important;
            right: 0 !important;
            left: auto !important;
            z-index: 1000000 !important;
            width: var(--agent-w) !important;
            max-width: 100vw !important;
            height: 100vh !important;
            height: 100dvh !important;
            box-sizing: border-box !important;
            padding: 0;
            display: flex;
            flex-direction: column;
            gap: 0 !important;
            overflow: hidden !important;
            min-width: 0 !important;
            color: #f4f4f5;
            background: #17191d;
            border-left: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: -12px 0 36px rgba(0, 0, 0, 0.28);
        }
        .st-key-agent_panel > [data-testid="stLayoutWrapper"] {
            flex: 0 0 auto !important;
            min-width: 0;
        }
        .st-key-agent_panel > [data-testid="stLayoutWrapper"]:has(.st-key-agent_messages) {
            flex: 1 1 0 !important;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }
        .st-key-agent_header,
        .st-key-agent_mode_switcher {
            flex: 0 0 auto !important;
            min-height: 0;
        }
        .st-key-agent_header [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        .st-key-agent_messages {
            flex: 1 1 0;
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior: contain;
            scrollbar-width: thin;
            scrollbar-color: transparent transparent;
            padding: 12px 16px 16px;
        }
        .st-key-agent_messages [data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
            width: 100%;
            flex: 1 1 0;
            min-height: 0;
            gap: 8px;
        }
        .st-key-agent_messages::-webkit-scrollbar {
            width: 6px;
        }
        .st-key-agent_messages::-webkit-scrollbar-track {
            background: transparent;
            border-radius: 3px;
        }
        .st-key-agent_messages::-webkit-scrollbar-thumb {
            background: transparent;
            border-radius: 3px;
        }
        .st-key-agent_messages:hover {
            scrollbar-color: #626873 transparent;
        }
        .st-key-agent_messages:hover::-webkit-scrollbar-thumb {
            background: #626873;
        }
        .st-key-agent_messages [data-testid="stMarkdownContainer"],
        .st-key-agent_messages [data-testid="stChatMessage"],
        .st-key-agent_messages p,
        .st-key-agent_messages li,
        .st-key-agent_messages span {
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .st-key-agent_messages table,
        .st-key-agent_messages pre,
        .st-key-agent_messages code,
        .st-key-agent_messages [data-testid="stCode"] {
            max-width: 100%;
            overflow-wrap: anywhere;
        }
        .st-key-agent_panel [data-testid="stForm"] {
            flex: 0 0 auto;
            box-sizing: border-box;
            width: 100%;
            padding: 12px 16px 16px;
            background: #17191d;
            border-top: 1px solid rgba(255, 255, 255, 0.14);
            border-right: 0;
            border-bottom: 0;
            border-left: 0;
            box-shadow: none;
        }
        .st-key-agent_panel [data-testid="stForm"] > [data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        .st-key-agent_panel [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 8px;
        }
        .st-key-agent_panel [data-testid="stForm"] [data-testid="stColumn"] {
            min-width: 0 !important;
            padding: 0 !important;
        }
        .st-key-agent_panel [data-testid="stForm"] [data-testid="stColumn"]:last-child {
            flex: 0 0 36px !important;
            width: 36px !important;
        }
        .st-key-agent_question_text {
            margin: 0;
        }
        .st-key-agent_question_text input {
            min-height: 40px;
            border-radius: 12px;
            background: #22252b;
            border-color: rgba(255, 255, 255, 0.14);
        }
        .st-key-FormSubmitter-agent_chat_input-Send button {
            width: 36px;
            height: 36px;
            min-height: 36px;
            padding: 0;
            border-radius: 50%;
            color: white;
            background: #e53935;
            border: none;
            font-size: 0;
        }
        .st-key-FormSubmitter-agent_chat_input-Send button::after {
            content: "➤";
            font-size: 16px;
        }
        .st-key-FormSubmitter-agent_chat_input-Send button:hover {
            background: #c62828;
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
            box-sizing: border-box;
            padding: 16px 16px 0;
        }
        .st-key-agent_header [data-testid="stHorizontalBlock"] {
            width: 100%;
            align-items: center;
            gap: 8px;
        }
        .st-key-agent_header [data-testid="stColumn"] {
            min-width: 0 !important;
            padding: 0 !important;
        }
        .st-key-agent_header [data-testid="stColumn"]:first-child {
            flex: 1 1 0 !important;
            width: auto !important;
        }
        .st-key-agent_header [data-testid="stColumn"]:nth-child(2),
        .st-key-agent_header [data-testid="stColumn"]:nth-child(3) {
            flex: 0 0 32px !important;
            width: 32px !important;
        }
        .st-key-agent_header [data-testid="stColumn"] [data-testid="stMarkdownContainer"] {
            min-width: 0;
        }
        .st-key-agent_header h3 {
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: #f4f4f5 !important;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .st-key-agent_header [data-testid="stButton"] button {
            width: 32px;
            height: 32px;
            min-height: 32px;
            padding: 0;
            border-radius: 8px;
            font-size: 15px;
            line-height: 1;
        }
        .st-key-agent_mode_switcher {
            flex: 0 0 auto;
            box-sizing: border-box;
            width: calc(100% - 32px);
            margin: 12px 16px 0;
        }
        .st-key-agent_mode_switcher [data-testid="stButtonGroup"] {
            width: 100%;
            min-height: 36px !important;
            display: flex;
        }
        .st-key-agent_mode_switcher [data-testid="stButtonGroup"] button {
            min-height: 36px !important;
            flex: 1 1 0;
            padding: 6px 8px;
            border-radius: 8px;
        }
        .st-key-agent_mode_switcher [data-testid="stButtonGroup"] button[aria-checked="true"] {
            color: #ffffff !important;
            background: #e53935 !important;
            border-color: #e53935 !important;
        }
        .st-key-agent_suggestions [data-testid="stVerticalBlock"] {
            gap: 8px;
        }
        .st-key-agent_suggestions button {
            width: 100%;
            height: auto;
            min-height: 0;
            padding: 10px 12px;
            white-space: normal;
            text-align: left;
            justify-content: flex-start;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            background: #202329;
            line-height: 1.35;
        }
        .st-key-agent_suggestions button:hover {
            background: #2a2d34;
            border-color: rgba(229, 57, 53, 0.55);
        }
        .st-key-agent_messages [data-testid="stChatMessage"] {
            align-items: flex-start;
            gap: 8px;
            padding: 0;
            background: transparent;
            font-size: 14px;
            line-height: 1.5;
        }
        .st-key-agent_messages [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            min-width: 0;
            max-width: 100%;
            font-size: 14px;
            line-height: 1.5;
        }
        .st-key-agent_messages [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
            width: 24px;
            height: 24px;
            flex: 0 0 auto;
            font-size: 14px;
        }
        .st-key-agent_messages [data-testid="stChatMessageAvatar"] img {
            width: 24px;
            height: 24px;
        }
        .st-key-agent_messages [data-testid="stChatMessage"][data-message-author="user"] [data-testid="stChatMessageAvatar"] {
            display: none;
        }
        .st-key-agent_messages [data-testid="stChatMessage"][data-message-author="user"] [data-testid="stMarkdownContainer"] {
            margin-left: auto;
            max-width: 88%;
            padding: 9px 12px;
            border-radius: 12px 12px 4px 12px;
            background: rgba(229, 57, 53, 0.18);
        }
        .st-key-agent_messages [data-testid="stChatMessage"] [data-testid="stButton"] button {
            min-height: 28px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
        }
        .st-key-agent_panel iframe[title="st.components.v1.html"] {
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            border: none !important;
        }
        @media (max-width: 899px) {
            [data-testid="stAppViewContainer"] {
                width: 100vw !important;
                max-width: 100vw !important;
            }
            [data-testid="stHeader"],
            [data-testid="stBottom"] {
                width: 100vw !important;
                max-width: 100vw !important;
            }
            .st-key-agent_panel {
                width: var(--agent-w);
                max-width: 100vw;
                padding-left: 14px;
                padding-right: 14px;
            }
        }
        """

    return f"""
    <style>
        .stAppViewContainer,
        [data-testid="stAppViewContainer"] {{
            width: 100vw !important;
            max-width: 100vw !important;
            transition: width 0.2s ease, max-width 0.2s ease;
        }}
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
        with st.chat_message(
            message["role"],
            avatar="🤖" if message["role"] == "assistant" else None,
        ):
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

    with st.chat_message("assistant", avatar="🤖"):
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
        const messages = window.parent.document.querySelector(
            ".st-key-agent_messages"
        );
        if (messages) {
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    messages.scrollTop = messages.scrollHeight;
                });
            });
        }
        </script>
        """,
        height=0,
    )


def _dispatch_layout_resize() -> None:
    st.components.v1.html(
        """
        <script>
        [100, 300, 700].forEach((delay) => {
            window.setTimeout(() => {
                window.parent.dispatchEvent(new Event("resize"));
            }, delay);
        });
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
    was_open = st.session_state.get("agent_panel_was_open", False)
    st.markdown(_styles(is_open), unsafe_allow_html=True)

    if is_open != was_open:
        _dispatch_layout_resize()

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

    should_scroll = False
    with st.container(key="agent_panel"):
        with st.container(key="agent_header"):
            title, minimize_column, close_column = st.columns([7, 1, 1])
            title.markdown("### Chart & Demand Assistant")
            minimize_column.button(
                "—",
                key="agent_minimize_button",
                help="Minimize (keeps your chat)",
                on_click=_minimize_agent,
            )
            close_column.button(
                "✕",
                key="agent_close_button",
                help="Close and clear chat",
                on_click=_close_agent,
                args=(_default_mode(active_tab),),
            )

        st.session_state["agent_mode_selector"] = st.session_state["agent_mode"]
        with st.container(key="agent_mode_switcher"):
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
        previous_mode = st.session_state.get("agent_last_rendered_mode")
        if (not was_open or previous_mode != mode) and _history_for_mode(mode):
            should_scroll = True
        st.session_state["agent_last_rendered_mode"] = mode
        messages_container = st.container(key="agent_messages")
        with messages_container:
            _render_history(mode)

        with st.form(key="agent_chat_input", clear_on_submit=True):
            input_column, send_column = st.columns([10, 1], gap="small")
            with input_column:
                question = st.text_input(
                    "Ask about charts, analytics, or demand",
                    key="agent_question_text",
                    label_visibility="collapsed",
                    placeholder="Ask about charts, analytics, or demand",
                )
            with send_column:
                submitted = st.form_submit_button(
                    "Send",
                    help="Send message",
                    use_container_width=True,
                )

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
