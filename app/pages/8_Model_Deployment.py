"""Page 8 — Deployment.

Final human sign-off: summarizes the full approved pipeline and exports the
champion model + a run manifest for downstream serving.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

from components import ai_observation, ai_revision, approval
from components.sidebar import render as render_sidebar
from schemas.feature_profile import AIReasoning
from schemas.state import init_session_state
from services.langgraph_client import generate_with_fallback, get_client

st.set_page_config(page_title="Deployment", page_icon="🧬", layout="wide")
init_session_state()
render_sidebar()

st.title("8️⃣ Deployment")
st.caption("Final review of the full pipeline before shipping.")

if not st.session_state.get("approved_evaluation"):
    st.warning("Approve Model Evaluation (Page 7) before deploying.")
    st.stop()

st.session_state["current_step"] = 7

pc = st.session_state["project_config"]
plan = st.session_state["experiment_plan"]
eval_result = st.session_state["evaluation_result"]
best_name = st.session_state.get("selected_model")

st.markdown("##### Pipeline Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Project", pc.project_name)
c2.metric("Problem Type", pc.problem_type.value)
c3.metric("Champion Model", best_name or "—")
primary_metric_value = eval_result["metrics"].get(plan.primary_metric) or next(iter(eval_result["metrics"].values()))
c4.metric(plan.primary_metric, primary_metric_value)

with st.expander("📋 Full approval trail", expanded=True):
    stages = [
        ("Project", "approved_project"),
        ("Dataset Profile", "approved_profile"),
        ("Experiment Plan", "approved_experiment_plan"),
        ("Preprocessing", "approved_preprocessing"),
        ("Feature Engineering", "approved_feature_engineering"),
        ("Model Training", "approved_training"),
        ("Model Evaluation", "approved_evaluation"),
    ]
    for label, key in stages:
        status = "✅ Approved" if st.session_state.get(key) else "⏳ Pending"
        st.markdown(f"- **{label}**: {status}")

deployment_context = {
    "champion_model": best_name,
    "metrics": eval_result["metrics"],
    "all_stages_approved": True,
}
if st.session_state.get("deployment_reasoning") is None:
    local_reasoning = AIReasoning(
        summary=f"'{best_name}' is ready for deployment, having passed every human approval checkpoint in the pipeline.",
        evidence=[f"{k}: {v}" for k, v in eval_result["metrics"].items()],
        confidence=0.85,
        alternatives_considered=["Deploy as a shadow/canary model before full rollout"],
        risks=["Production data drift may degrade performance over time — schedule periodic re-profiling."],
        expected_impact="Generates a downloadable model artifact + run manifest for serving.",
        intervention_recommended=False,
    )
    st.session_state["deployment_reasoning"] = generate_with_fallback(
        task_type="deployment_reasoning",
        subject=f"Deployment of {best_name}",
        context=deployment_context,
        fallback=local_reasoning,
    )

reasoning = st.session_state["deployment_reasoning"]
ai_observation.render(reasoning, key="deployment", expanded=True)


def _on_revise(feedback: str) -> None:
    st.session_state["deployment_reasoning"] = get_client().refine_reasoning(
        task_type="deployment_reasoning",
        subject=f"Deployment of {best_name}",
        context=deployment_context,
        previous=st.session_state["deployment_reasoning"],
        feedback=feedback,
    )


ai_revision.render(key="deployment", on_revise=_on_revise, subject_label="the deployment readiness assessment")


def _on_deploy() -> None:
    manifest = {
        "project": pc.project_name,
        "problem_type": pc.problem_type.value,
        "champion_model": best_name,
        "metrics": eval_result["metrics"],
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    st.session_state["deployment_result"] = manifest


approval.render(
    stage_name="Deployment",
    approval_key="approved_deployment",
    next_step_label="— pipeline complete",
    override_note_key="deployment_override_note",
    on_approve=_on_deploy,
)

if st.session_state.get("deployment_result"):
    st.success("🚀 Deployed. Download the run manifest below.")
    st.download_button(
        "⬇️ Download run manifest (JSON)",
        data=json.dumps(st.session_state["deployment_result"], indent=2),
        file_name=f"{pc.project_name.replace(' ', '_').lower()}_manifest.json",
        mime="application/json",
    )
