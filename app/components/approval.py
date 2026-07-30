"""The mandatory approval checkpoint. No stage advances automatically -
the data scientist must explicitly click Approve (optionally after
overriding the AI's recommendation)."""
from __future__ import annotations

from typing import Callable, Optional

import streamlit as st


def render(
    stage_name: str,
    approval_key: str,
    next_step_label: str,
    on_approve: Optional[Callable[[], None]] = None,
    allow_override: bool = True,
    override_note_key: Optional[str] = None,
) -> bool:

    st.divider()
    approved = st.session_state.get(approval_key, False)

    if approved:
        st.success(f"✅ {stage_name} approved. Proceed to **{next_step_label}** from the sidebar.")
        if st.button("Revoke approval", key=f"{approval_key}_revoke"):
            st.session_state[approval_key] = False
            st.rerun()
        return True

    if allow_override and override_note_key:
        with st.expander("✍️ Manual override / notes (optional)"):
            st.text_area(
                "Describe any manual changes you want to make. This can be taken under consideration while making Experiment plan and Preprocessing .",
                key=override_note_key,
                placeholder="e.g. Remove 'Cabin' Column as it has a large amount of missing values...",
            )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(f"✅ Approve {stage_name}", type="primary", width="stretch", key=f"{approval_key}_btn"):
            st.session_state[approval_key] = True
            if on_approve:
                on_approve()
            st.rerun()
    with col2:
        st.caption(f"Nothing proceeds automatically — approving unlocks **{next_step_label}**.")

    return False
