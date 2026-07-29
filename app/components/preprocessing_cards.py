"""Renders each preprocessing step as a reviewable card with its own AI
Reasoning panel, applies-to column list, and a revision loop so the data
scientist can send feedback back to the AI on any individual step."""
from __future__ import annotations

import streamlit as st

from components import ai_observation, ai_revision
from services.preprocessing_service import PreprocessingPlan, PreprocessingService


def render(plan: PreprocessingPlan, service: PreprocessingService, plan_session_key: str = "preprocessing_plan") -> None:
    if not plan.steps:
        st.info("No preprocessing steps were flagged — this dataset appears clean as-is.")
        return

    for i, step in enumerate(plan.steps):
        with st.container():
            st.markdown(
                f"""<div class="aa-card">
                    <h4>{i + 1}. {step.title}</h4>
                    <span class="aa-badge aa-badge-blue">{step.method}</span>
                    <span class="aa-badge aa-badge-gray">{len(step.applies_to)} column(s)</span>
                </div>""",
                unsafe_allow_html=True,
            )
            cols_preview = ", ".join(f"`{c}`" for c in step.applies_to[:10])
            more = f" … and {len(step.applies_to) - 10} more" if len(step.applies_to) > 10 else ""
            st.markdown(f"**Applies to:** {cols_preview}{more}")
            ai_observation.render(step.reasoning, key=f"preproc_{step.step_id}")

            def _on_revise(feedback: str, step_id: str = step.step_id) -> None:
                current_plan: PreprocessingPlan = st.session_state[plan_session_key]
                target = next(s for s in current_plan.steps if s.step_id == step_id)
                target.reasoning = service.refine_step_reasoning(target, feedback)
                st.session_state[plan_session_key] = current_plan

            ai_revision.render(key=f"preproc_{step.step_id}", on_revise=_on_revise, subject_label=f"the '{step.title}' step")
            st.write("")

