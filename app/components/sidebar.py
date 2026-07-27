import streamlit as st


from components.workflow_progress import render as render_workflow_progress
from schemas.state import reset_session_state

_CSS = """
<style>
    .block-container { padding-top: 2rem; max-width: 1300px; }
    #MainMenu, footer { visibility: hidden; }

    .aa-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.9rem;
    }
    .aa-card h4 { margin: 0 0 0.4rem 0; font-size: 0.95rem; color:#0F172A;}
    .aa-metric-label { font-size: 0.78rem; color:#64748B; text-transform: uppercase; letter-spacing:.04em;}
    .aa-metric-value { font-size: 1.5rem; font-weight: 700; color:#0F172A;}

    .aa-badge {
        display:inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size:0.72rem; font-weight:600; letter-spacing:.02em;
    }
    .aa-badge-green { background:#DCFCE7; color:#166534; }
    .aa-badge-amber { background:#FEF3C7; color:#92400E; }
    .aa-badge-red { background:#FEE2E2; color:#991B1B; }
    .aa-badge-blue { background:#DBEAFE; color:#1E40AF; }
    .aa-badge-gray { background:#F1F5F9; color:#334155; }

    .aa-pill-nav { display:flex; gap:0.5rem; overflow-x:auto; padding: 0.4rem 0.1rem 0.9rem 0.1rem; }
    .aa-pill-nav::-webkit-scrollbar { height: 6px; }
    .aa-pill-nav::-webkit-scrollbar-thumb { background:#CBD5E1; border-radius: 4px; }

    div[data-testid="stSidebarUserContent"] { padding-top: 1rem; }

    .aa-step-done { color:#16A34A; font-weight:600; }
    .aa-step-active { color:#4F46E5; font-weight:700; }
    .aa-step-pending { color:#94A3B8; }
</style>
"""

def inject_global_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

def render() -> None:
    """Render the sidebar: project identity, workflow checklist, reset."""
    inject_global_css()
    with st.sidebar:
        project_config = st.session_state.get("project_config")
        st.markdown("### 🧬 Agentic AutoML")
        if project_config:
            st.caption(f"**{project_config.project_name}**")
            st.caption(f"{project_config.problem_type.value} · target: `{project_config.target_column or '—'}`")
        else:
            st.caption("No project configured yet")

        st.divider()
        st.markdown("##### Workflow")
        render_workflow_progress(st.session_state.get("current_step", 0))

        st.divider()
        if st.button("🔄 Reset Project", width="stretch"):
            reset_session_state()
            st.rerun()
