

import streamlit as st
from schemas.state import WORKFLOW_STEPS


def render(current_step: int) -> None:
    """Render the workflow checklist.

    Parameters
    ----------
    current_step:
        Zero-based index into `WORKFLOW_STEPS` for the step currently in
        progress. Earlier steps render as done, later steps as pending.
    """
    for i, step_name in enumerate(WORKFLOW_STEPS):
        if i < current_step:
            st.markdown(f'<span class="aa-step-done">✔ {step_name}</span>', unsafe_allow_html=True)
        elif i == current_step:
            st.markdown(f'<span class="aa-step-active">▶ {step_name}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="aa-step-pending">○ {step_name}</span>', unsafe_allow_html=True)