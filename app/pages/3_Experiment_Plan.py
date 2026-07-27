import streamlit as st

from app.state import initialize
from components.sidebar import render_sidebar
from services.experiment import get_experiment_plan

initialize()
st.set_page_config(page_title="Experiment Plan", layout="wide")
render_sidebar(current_page="Experiment Plan")

st.title("Experiment plan")
st.caption("Review the generated preprocessing, model, and evaluation decisions before execution.")

if st.session_state.project_config is None:
	st.warning("Set up a project before reviewing its experiment plan.")
	st.page_link("pages/1_Project.py", label="Go to project setup")
	st.stop()

plan = get_experiment_plan()
if plan is None:
	st.info("No generated experiment plan is available yet.")
	st.stop()

preprocessing = plan.get("preprocessing", {})
left, right = st.columns(2)
with left:
	st.subheader("Preprocessing")
	dtype_overrides = st.session_state.get("dataset_dtype_overrides", {})
	if dtype_overrides:
		st.write("**User dtype overrides:**")
		st.json(dtype_overrides)
	else:
		st.caption("No user dtype overrides recorded.")
	st.write("**Missing values:** " + ", ".join(preprocessing.get("missing_value_strategy", [])))
	st.write("**Encoding:** " + ", ".join(preprocessing.get("encoding_strategy", [])))
	st.write(f"**Scaling required:** {'Yes' if preprocessing.get('scaling_required') else 'No'}")
	st.write(f"**Feature selection:** {'Yes' if preprocessing.get('feature_selection_required') else 'No'}")
with right:
	st.subheader("Evaluation")
	evaluation = plan.get("evaluation", {})
	st.metric("Primary metric", str(evaluation.get("primary_metric", "")).upper())
	st.write("**Secondary metrics:** " + ", ".join(evaluation.get("secondary_metrics", [])))
	st.write("**Validation:** " + str(evaluation.get("validation_strategy", "")).replace("_", " ").title())

st.divider()
st.subheader("Recommended models")
for model in sorted(plan.get("models", []), key=lambda item: item.get("priority", 99)):
	with st.container(border=True):
		st.write(f"**{model.get('priority', '')}. {str(model.get('model_name', '')).replace('_', ' ').title()}**")
		st.caption(model.get("reason", ""))

st.divider()
st.subheader("Risks to monitor")
for risk in plan.get("risks", []):
	st.warning(str(risk).replace("_", " ").title())

st.session_state.experiment_plan = plan
st.session_state.approved_experiment = st.checkbox(
	"I have reviewed the experiment plan", value=st.session_state.approved_experiment
)
