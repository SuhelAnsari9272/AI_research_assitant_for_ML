"""The review-and-revise widget used on every page from Page 2 onward.

This is the actual human-in-the-loop mechanism: the data scientist reads the
AI's reasoning, optionally writes feedback, and clicking the button sends
that feedback back into the LangGraph agent (via the page's `on_revise`
callback), which returns a *new* `AIReasoning` reflecting the requested
change - not just a comment thread bolted onto a static recommendation.
"""
from __future__ import annotations

from typing import Callable

import streamlit as st

from services.langgraph_client import LangGraphUnavailable


def render(key: str, on_revise: Callable[[str], None], subject_label: str = "this recommendation") -> None:
    """Render the feedback box + 'send to AI' button for one AI decision.

    Parameters
    ----------
    key: Unique key prefix for this decision's widgets (e.g. feature name, step id).
    on_revise: Callback invoked with the feedback text; should call the
        relevant service's `refine_*` method and write the new `AIReasoning`
        back into session state.
    subject_label: Human-readable name of the thing being revised, shown in
        the expander title (e.g. "the 'salary' feature's classification").
    """
    with st.expander(f"💬 Request AI changes to {subject_label}", expanded=False):
        st.caption(
            "Disagree with the recommendation above? Explain what should change and the AI "
            "will re-reason about it - this doesn't just log a comment, it asks the LLM to "
            "actually revise its recommendation."
        )
        feedback = st.text_area(
            "Feedback for the AI",
            key=f"{key}_feedback_text",
            placeholder="e.g. 'zip_code' should be Categorical, not Numerical - it's not ordinal.",
            label_visibility="collapsed",
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            send = st.button("🔁 Send to AI", key=f"{key}_revise_btn", disabled=not feedback.strip())
        if send:
            try:
                with st.spinner("Sending feedback to the reasoning agent..."):
                    on_revise(feedback.strip())
                st.success("Recommendation updated based on your feedback.")
                st.rerun()
            except LangGraphUnavailable as exc:
                st.error(
                    f"Couldn't reach the AI reasoning agent ({exc}). Check `GROQ_API_KEY` in your "
                    "environment - see .env.example."
                )
