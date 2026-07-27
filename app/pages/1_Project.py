from __future__ import annotations

import streamlit as st
import pandas as pd

from components.sample_dataset import render as render_sample
from components.sidebar import render as render_sidebar
from schemas.state import ProblemType, ProjectConfig, init_session_state
from services.profiler import DatasetProfilerService
from utils.formatting import format_bytes, format_number

st.set_page_config(page_title="Project Configuration", page_icon="🧬", layout="wide")
init_session_state()
render_sidebar()

st.title("1️⃣ Project Configuration")
st.caption("Define the problem, then upload the dataset the AI will profile.")

# Project metadata
# --------------------------------------------------------------------------- #
existing: ProjectConfig = st.session_state.get("project_config") or ProjectConfig()

with st.container():
    st.markdown("##### Project Details")
    c1, c2 = st.columns(2)
    with c1:
        project_name = st.text_input("Project Name", value=existing.project_name, placeholder="e.g. Customer Churn Predictor")
        problem_type = st.selectbox(
            "Problem Type",
            options=list(ProblemType),
            index=list(ProblemType).index(existing.problem_type),
            format_func=lambda p: p.value,
        )
    with c2:
        evaluation_metric = st.text_input(
            "Evaluation Metric",
            value=existing.evaluation_metric,
            placeholder="e.g. ROC-AUC, RMSE, F1",
        )
        business_goal = st.text_input(
            "Business Goal",
            value=existing.business_goal,
            placeholder="e.g. Reduce customer churn by identifying at-risk accounts",
        )

st.divider()

# Dataset upload
# --------------------------------------------------------------------------- #
st.markdown("##### Upload Dataset")
uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

df: pd.DataFrame | None = st.session_state.get("dataset")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["dataset"] = df
        st.session_state["dataset_name"] = uploaded_file.name
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read this CSV: {exc}")
        df = None

target_column = existing.target_column
if df is not None:
    st.success(f"Loaded **{st.session_state.get('dataset_name', 'dataset')}** — "
               f"{format_number(df.shape[0])} rows × {format_number(df.shape[1])} columns "
               f"({format_bytes(df.memory_usage(deep=True).sum())})")

    st.markdown("##### Dataset Preview")
    render_sample(df)

    st.markdown("##### Dataset Statistics")
    stat_cols = st.columns(4)
    stat_cols[0].metric("Rows", format_number(df.shape[0]))
    stat_cols[1].metric("Columns", format_number(df.shape[1]))
    stat_cols[2].metric("Missing Cells", format_number(int(df.isna().sum().sum())))
    stat_cols[3].metric("Duplicate Rows", format_number(int(df.duplicated().sum())))

    st.markdown("##### Target Column")
    target_column = st.selectbox(
        "Select the column the model should predict",
        options=list(df.columns),
        index=list(df.columns).index(existing.target_column) if existing.target_column in df.columns else 0,
    )
else:
    st.info("Upload a CSV to continue. A target column can be selected once data is loaded.")

st.divider()

# Save + generate profile
# --------------------------------------------------------------------------- #
col_save, col_profile = st.columns([1, 1])

with col_save:
    if st.button("💾 Save Configuration", width="stretch", disabled=not project_name):
        st.session_state["project_config"] = ProjectConfig(
            project_name=project_name,
            problem_type=problem_type,
            target_column=target_column,
            evaluation_metric=evaluation_metric,
            business_goal=business_goal,
        )
        st.session_state["approved_project"] = True
        st.session_state["current_step"] = 1
        st.success("Project configuration saved.")

with col_profile:
    ready = st.session_state.get("project_config") is not None and df is not None
    if st.button("🚀 Generate Dataset Profile", type="primary", width="stretch", disabled=not ready):
        with st.spinner("Profiling dataset — inferring types, computing statistics, generating AI reasoning..."):
            service = DatasetProfilerService()
            profile = service.profile_dataset(
                df=df,
                target_column=target_column,
                problem_type=problem_type.value,
            )
            st.session_state["dataset_profile"] = profile
            st.session_state["current_step"] = 1
        st.success("Dataset profile generated — open **2_Dataset_Profile** from the sidebar.")