
from __future__ import annotations

import streamlit as st

from schemas.feature_profile import AIReasoning


def render(reasoning: AIReasoning, expanded: bool = False, key: str = "") -> None:
    """Render the AI Reasoning expander for a single decision."""

    with st.expander(f"🧠 Reasoning -", expanded=expanded):
        st.markdown(f"**Why**  \n{reasoning.summary}")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Evidence used**")
            for e in reasoning.evidence or ["—"]:
                st.markdown(f"- {e}")
            st.markdown("**Alternatives considered**")
            for a in reasoning.alternatives_considered or ["—"]:
                st.markdown(f"- {a}")
        with cols[1]:
            st.markdown("**Potential risks**")
            for r in reasoning.risks or ["—"]:
                st.markdown(f"- {r}")
            st.markdown("**Expected impact if approved**")
            st.markdown(reasoning.expected_impact)

        if reasoning.intervention_recommended:
            st.warning("The AI recommends a human double-check this decision before approving.")
