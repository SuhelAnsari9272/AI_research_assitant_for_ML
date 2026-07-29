"""Page 5 — Feature Engineering.

Applies the approved preprocessing plan to materialize the model-ready
feature matrix, and proposes a small set of derived features (e.g. log
transforms for skewed columns) with before/after previews.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from components import ai_observation, ai_revision, approval
from components.sidebar import render as render_sidebar
from schemas.feature_profile import AIReasoning, FeatureRole
from schemas.state import init_session_state
from services.langgraph_client import get_client, generate_with_fallback

st.set_page_config(page_title="Feature Engineering", page_icon="🧬", layout="wide")
init_session_state()
render_sidebar()

st.title("5️⃣ Feature Engineering")
st.caption("Applying the approved preprocessing plan to build the model-ready feature matrix.")

profile = st.session_state.get("dataset_profile")
df = st.session_state.get("dataset")
preproc_plan = st.session_state.get("preprocessing_plan")

if profile is None or df is None or not st.session_state.get("approved_preprocessing"):
    st.warning("Approve the Preprocessing Plan (Page 4) before engineering features.")
    st.stop()

st.session_state["current_step"] = 4


def _build_feature_matrix(df: pd.DataFrame, profile) -> pd.DataFrame:
    working = df.copy()
    drop_cols = [f.name for f in profile.features.values() if f.role == FeatureRole.IDENTIFIER or f.leakage_risk]
    working = working.drop(columns=[c for c in drop_cols if c in working.columns], errors="ignore")

    numeric_cols, skewed_cols, categorical_cols = [], [], []
    for f in profile.features.values():
        if f.name not in working.columns or f.role != FeatureRole.FEATURE:
            continue
        if f.semantic_type.value == "Numerical":
            numeric_cols.append(f.name)
            if f.skewness is not None and abs(f.skewness) >= 0.5:
                skewed_cols.append(f.name)
        elif f.semantic_type.value in ("Categorical", "Ordinal", "Binary"):
            categorical_cols.append(f.name)

    for col in numeric_cols:
        working[col] = working[col].fillna(working[col].median())
    for col in categorical_cols:
        mode = working[col].mode(dropna=True)
        working[col] = working[col].fillna(mode.iloc[0] if not mode.empty else "missing")

    for col in skewed_cols:
        working[f"{col}_log"] = np.log1p(working[col].clip(lower=0))

    if numeric_cols:
        scaler = StandardScaler()
        working[numeric_cols] = scaler.fit_transform(working[numeric_cols])

    if categorical_cols:
        low_card = [c for c in categorical_cols if working[c].nunique() <= 15]
        high_card = [c for c in categorical_cols if c not in low_card]
        if low_card:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = ohe.fit_transform(working[low_card].astype(str))
            encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(low_card), index=working.index)
            working = working.drop(columns=low_card).join(encoded_df)
        for c in high_card:
            freq = working[c].value_counts(normalize=True)
            working[f"{c}_freq_enc"] = working[c].map(freq)
            working = working.drop(columns=[c])

    return working, skewed_cols, numeric_cols, categorical_cols


if st.session_state.get("feature_engineering_plan") is None:
    with st.spinner("Engineering features..."):
        engineered_df, skewed_cols, numeric_cols, categorical_cols = _build_feature_matrix(df, profile)
        context = {
            "numeric_feature_count": len(numeric_cols),
            "categorical_feature_count": len(categorical_cols),
            "skewed_feature_count": len(skewed_cols),
            "skewed_features": skewed_cols,
            "resulting_matrix_shape": list(engineered_df.shape),
        }
        local_reasoning = AIReasoning(
            summary=(
                f"Built the model-ready matrix: imputed {len(numeric_cols)} numeric + "
                f"{len(categorical_cols)} categorical columns, log-transformed {len(skewed_cols)} skewed "
                f"column(s), scaled numerics, and encoded categoricals."
            ),
            evidence=[
                f"{len(numeric_cols)} numeric feature(s) detected.",
                f"{len(categorical_cols)} categorical feature(s) detected.",
                f"{len(skewed_cols)} feature(s) exceeded |skewness| >= 0.5.",
            ],
            confidence=0.83,
            alternatives_considered=["Leave categoricals unencoded for tree models only", "Target encoding instead of one-hot"],
            risks=["One-hot encoding increased dimensionality; watch for sparsity with small datasets."],
            expected_impact="This exact matrix (minus the target) is what gets passed into Model Training.",
            intervention_recommended=False,
        )
        reasoning = generate_with_fallback(
            task_type="feature_engineering_reasoning",
            subject="Feature engineering plan",
            context=context,
            fallback=local_reasoning,
        )
        st.session_state["feature_engineering_plan"] = {
            "engineered_df": engineered_df,
            "reasoning": reasoning,
            "context": context,
            "skewed_cols": skewed_cols,
        }

plan = st.session_state["feature_engineering_plan"]
engineered_df: pd.DataFrame = plan["engineered_df"]

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Before")
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(8), width="stretch")
with c2:
    st.markdown("##### After (model-ready)")
    st.caption(f"{engineered_df.shape[0]} rows × {engineered_df.shape[1]} columns")
    st.dataframe(engineered_df.head(8), width="stretch")

if plan["skewed_cols"]:
    st.info(f"Added log-transformed columns for: {', '.join(f'`{c}_log`' for c in plan['skewed_cols'])}")

ai_observation.render(plan["reasoning"], key="feature_engineering", expanded=True)


def _on_revise(feedback: str) -> None:
    updated = get_client().refine_reasoning(
        task_type="feature_engineering_reasoning",
        subject="Feature engineering plan",
        context=plan["context"],
        previous=plan["reasoning"],
        feedback=feedback,
    )
    st.session_state["feature_engineering_plan"]["reasoning"] = updated


ai_revision.render(key="feature_engineering", on_revise=_on_revise, subject_label="the feature engineering plan")

approval.render(
    stage_name="Feature Engineering",
    approval_key="approved_feature_engineering",
    next_step_label="Model Training",
    override_note_key="fe_override_note",
    on_approve=lambda: st.session_state.update(current_step=5),
)
