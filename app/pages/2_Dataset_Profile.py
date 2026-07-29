from __future__ import annotations
import streamlit as st

from components import ai_observation, ai_revision, approval, distribution, feature_card, feature_navigator
from components.sample_dataset import render as render_sample
from components.sidebar import render as render_sidebar
from components.summary_cards import render as render_summary_cards
from schemas.state import init_session_state
from services.langgraph_client import LangGraphUnavailable, get_client
from services.profiler import DatasetProfilerService
from utils import plot_utils


st.set_page_config(page_title="Dataset Profile", page_icon="🧬", layout="wide")
init_session_state()
render_sidebar()

st.title("2️⃣ Dataset Profile")
st.caption("Inspect every feature individually. Nothing here is a giant dataframe.")

profile = st.session_state.get("dataset_profile")
df = st.session_state.get("dataset")

if profile is None or df is None:
    st.warning("No dataset profile yet. Go to **1_Project**, upload a dataset, and click 'Generate Dataset Profile'.")
    st.stop()

st.session_state["current_step"] = 1

llm_client = get_client()
profiler_service = DatasetProfilerService()

# Summary cards
# --------------------------------------------------------------------------- #
render_summary_cards(profile)
st.divider()


# Data preview + missingness overview
# --------------------------------------------------------------------------- #
with st.expander("📄 Data Preview (Head / Tail / Random Sample)", expanded=False):
    render_sample(df)

with st.expander("🕳️ Missingness Overview", expanded=False):
    missing_pct_by_col = (df.isna().mean() * 100).round(2)
    st.plotly_chart(plot_utils.missingness_bar(missing_pct_by_col), width="stretch", config={"displayModeBar": False})

st.divider()

# Feature Explorer
# --------------------------------------------------------------------------- #
st.markdown("### 🔬 Feature Explorer")

feature_names = list(profile.features.keys())
roles = {name: fp.role for name, fp in profile.features.items()}

selected = feature_navigator.render(
    feature_names=feature_names,
    selected=st.session_state.get("selected_feature"),
    roles=roles,
)
st.session_state["selected_feature"] = selected

active_feature = profile.features[selected]

left, right = st.columns([2, 1])
with left:
    feature_card.render(active_feature)
with right:
    distribution.render(df[selected], active_feature)

# --------------------------------------------------------

# AI Reasoning: local heuristic by default, upgradeable to a live Groq call,
# then revisable through the feedback loop.
# --------------------------------------------------------------------------- #
llm_reasoning_cache = st.session_state.setdefault("feature_llm_reasoning", {})
is_llm_backed = selected in llm_reasoning_cache
current_reasoning = llm_reasoning_cache.get(selected, active_feature.reasoning)

badge_col, button_col = st.columns([4, 1])
with badge_col:
    st.caption("🤖 Live AI reasoning" if is_llm_backed else "📐 Local heuristic reasoning")
with button_col:
    if llm_client.is_configured and st.button("🤖 Ask AI", key=f"llm_gen_{selected}", width="stretch"):
        try:
            with st.spinner(f"Asking Groq to reason about '{selected}'..."):
                llm_reasoning_cache[selected] = profiler_service.generate_llm_reasoning(active_feature)
            st.rerun()
        except LangGraphUnavailable as exc:
            st.error(f"AI reasoning call failed, showing local heuristic instead: {exc}")

ai_observation.render(current_reasoning, key=f"feature_{selected}")

def _on_revise(feedback: str) -> None:
    updated = profiler_service.refine_llm_reasoning(active_feature, feedback)
    st.session_state["feature_llm_reasoning"][selected] = updated


ai_revision.render(key=f"feature_{selected}", on_revise=_on_revise, subject_label=f"the `{selected}` classification")

# --------------------------------------------------------------------------- #
# Approval checkpoint
# --------------------------------------------------------------------------- #
approval.render(
    stage_name="Dataset Profile",
    approval_key="approved_profile",
    next_step_label="Experiment Plan",
    override_note_key="profile_override_note",
    on_approve=lambda: st.session_state.update(current_step=2),
)