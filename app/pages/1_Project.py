import streamlit as st
import io
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from schemas.project_config import ProjectConfig
from app.state import initialize
from app.services.langgraph import execute_langgraph_workflow

initialize()

st.title("Project Configuration")

left, right = st.columns([2,1])
project_name = left.text_input("Project Name")
objective = left.text_input("Objective")
problem_type = left.selectbox("Problem Type", 
                              ["Classification", "Regression"]
                              )
target_column = left.text_input("Target Column")
metric = left.selectbox( "Optimization Metric", ["Accuracy", "F1", "ROC AUC", "RMSE", "MAE"])
goal = left.text_area("Business Goal", height=150)

if left.button("Save Project Configuration"):

    st.session_state.project_config = ProjectConfig(
        project_name=project_name,
        objective=objective, 
        target_column=target_column,
        problem_type=problem_type,
        preferred_metric=metric   
    )

    st.session_state.dataset_profile = None
    st.session_state.experiment_plan = None

    st.success("Configuration Saved")

with right:
    st.subheader("Current Config")
    st.json(st.session_state.project_config)

st.divider()
st.header("Upload Dataset")

# Upload Dataset
uploaded = st.file_uploader( "Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state.dataset = df
    st.session_state.dataset_profile = None

# Dataset Preview
if st.session_state.dataset is not None:
    st.subheader("Preview")
    st.dataframe(st.session_state.dataset.head(), use_container_width=True )

# Dataset Statistics
if st.session_state.dataset is not None:

    df = st.session_state.dataset

    buffer = io.StringIO()
    df.info(buf=buffer)

    st.code(buffer.getvalue(), language="text")


# continue
if st.session_state.dataset is not None:

    if st.button("Generate Dataset Profile"):
        st.switch_page("pages/2_Dataset_Profile.py")