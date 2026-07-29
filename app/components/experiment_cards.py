"""Renders candidate model cards for the Experiment Planning stage."""
from __future__ import annotations

import streamlit as st

from services.experiment_service import ExperimentPlan


def render(plan: ExperimentPlan) -> None:
    st.markdown("##### Candidate Models")
    cols = st.columns(len(plan.candidate_models))
    for col, model in zip(cols, plan.candidate_models):
        with col:
            st.markdown(
                f"""<div class="aa-card">
                    <h4>{model.name}</h4>
                    <span class="aa-badge aa-badge-blue">{model.library}</span>
                    <p style="margin-top:0.6rem; font-size:0.86rem; color:#334155;">{model.rationale}</p>
                    <p style="font-size:0.78rem; color:#64748B;"><b>Strength:</b> {model.expected_strength}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("##### Validation Strategy")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="aa-card"><div class="aa-metric-label">Primary Metric</div>'
                f'<div class="aa-metric-value" style="font-size:1.1rem;">{plan.primary_metric}</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="aa-card"><div class="aa-metric-label">Validation</div>'
                f'<div class="aa-metric-value" style="font-size:1.1rem;">{plan.validation_strategy}</div></div>',
                unsafe_allow_html=True)
