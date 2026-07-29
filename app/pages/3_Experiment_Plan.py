from __future__ import annotations
import streamlit as st

from components import ai_observation, ai_revision, approval, experiment_cards
from components.sidebar import render as render_sidebar
from schemas.state import init_session_state
from services.experiment_service import ExperimentService

st.set_page_config(page_title="Experiment Plan", page_icon="🧬", layout="wide")
init_session_state()
render_sidebar()

st.title("3️⃣ Experiment Planning")
st.caption("AI-proposed models, metric, and validation strategy — reviewable before any training happens.")

profile = st.session_state.get("dataset_profile")
if profile is None or not st.session_state.get("approved_profile"):
    st.warning("Approve the Dataset Profile (Page 2) before planning the experiment.")
    st.stop()

st.session_state["current_step"] = 2

service = ExperimentService()

if st.session_state.get("experiment_plan") is None:
    with st.spinner("Designing experiment plan..."):
        pc = st.session_state.get("project_config")
        st.session_state["experiment_plan"] = service.build_plan(
            profile, evaluation_metric_hint=pc.evaluation_metric if pc else ""
        )

plan = st.session_state["experiment_plan"]

experiment_cards.render(plan)
ai_observation.render(plan.reasoning, key="experiment_plan", expanded=True)

def _on_revise(feedback: str) -> None:
    plan.reasoning = service.refine_plan_reasoning(plan, profile, feedback)
    st.session_state["experiment_plan"] = plan


ai_revision.render(key="experiment_plan", on_revise=_on_revise, subject_label="the experiment plan")

if st.button("🔁 Regenerate plan from scratch"):
    st.session_state["experiment_plan"] = None
    st.rerun()

approval.render(
    stage_name="Experiment Plan",
    approval_key="approved_experiment_plan",
    next_step_label="Preprocessing",
    override_note_key="experiment_override_note",
    on_approve=lambda: st.session_state.update(current_step=3),
)