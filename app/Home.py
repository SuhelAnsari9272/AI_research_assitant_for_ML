import streamlit as st
from state import initialize
from components.sidebar import render_sidebar

initialize()

st.set_page_config(
    page_title="AutoML Agent",
    layout="wide"
)

render_sidebar(current_page="Workspace overview")
st.title("Agentic AutoML")

st.markdown(
"""
Use the workspace to move from project context to dataset evidence and an
experiment plan that you can review before training.
"""
)

config = st.session_state.project_config
dataset = st.session_state.dataset
profile = st.session_state.dataset_profile
plan = st.session_state.experiment_plan

st.divider()
st.subheader("Workspace status")
status = st.columns(4)
status[0].metric("Project", "Ready" if config else "Not set")
status[1].metric("Dataset", f"{len(dataset):,} rows" if dataset is not None else "Not uploaded")
status[2].metric("Profile", "Ready" if profile else "Pending")
status[3].metric("Plan", "Ready" if plan else "Available in artifacts")

st.page_link("pages/1_Project.py", label="Open project setup")