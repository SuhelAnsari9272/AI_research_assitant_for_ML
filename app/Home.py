from __future__ import annotations
import streamlit as st
from components.sidebar import render as render_sidebar
from schemas.state import init_session_state


st.set_page_config(
    page_title="Agentic AutoML Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar()

st.title("🧬 Agentic AutoML Platform")
st.caption("A human-in-the-loop machine learning workbench — the AI reasons, you decide.")

st.markdown(
    """
    <div class="aa-card">
    <h4>How this platform works</h4>
    <p style="color:#334155; font-size:0.92rem;">
    At every stage the AI proposes a decision — a data type, a preprocessing step, a model,
    a metric — and shows you <b>why</b>: the evidence it used, its confidence, the alternatives
    it considered, and the risks involved. Nothing advances to the next stage until you
    explicitly click <b>Approve</b>, and you can always override the AI's recommendation first.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
cols = st.columns(4)
stage_blurbs = [
    ("1️⃣ Project & Dataset", "Define the problem and upload your data."),
    ("2️⃣ Dataset Profile", "Inspect every feature: type, stats, risks."),
    ("3️⃣–5️⃣ Planning", "Approve the experiment, preprocessing & feature plans."),
    ("6️⃣–9️⃣ Train & Ship", "Train, evaluate, and deploy — with sign-off at each step."),
]
for c, (title, blurb) in zip(cols, stage_blurbs):
    with c:
        st.markdown(
            f'<div class="aa-card" style="min-height:120px;"><h4>{title}</h4>'
            f'<p style="font-size:0.85rem;color:#475569;">{blurb}</p></div>',
            unsafe_allow_html=True,
        )

st.write("")
if st.session_state.get("project_config") is None:
    st.info("👉 Start in **1_Project** from the sidebar's Pages menu to configure your first project.")
else:
    pc = st.session_state["project_config"]
    st.success(f"Active project: **{pc.project_name}** — continue where you left off using the sidebar.")