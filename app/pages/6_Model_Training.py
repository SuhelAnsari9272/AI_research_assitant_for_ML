"""Page 6 — Model Training.

Actually trains the candidate models proposed in the Experiment Plan against
the engineered feature matrix, using the approved validation strategy.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from components import ai_observation, ai_revision, approval
from components.sidebar import render as render_sidebar
from schemas.feature_profile import AIReasoning
from schemas.state import init_session_state
from services.langgraph_client import generate_with_fallback, get_client
from utils import plot_utils

st.set_page_config(page_title="Model Training", page_icon="🧬", layout="wide")
init_session_state()
render_sidebar()

st.title("6️⃣ Model Training")
st.caption("Training each approved candidate model and cross-validating on the approved metric.")

profile = st.session_state.get("dataset_profile")
plan = st.session_state.get("experiment_plan")
fe_plan = st.session_state.get("feature_engineering_plan")

if not all([profile, plan, fe_plan]) or not st.session_state.get("approved_feature_engineering"):
    st.warning("Approve Feature Engineering (Page 5) before training models.")
    st.stop()

st.session_state["current_step"] = 5

is_classification = plan.problem_type.lower().startswith("class")
target_col = profile.target_column
engineered_df: pd.DataFrame = fe_plan["engineered_df"]

if target_col not in engineered_df.columns:
    st.error(f"Target column '{target_col}' not found in engineered matrix. Check Feature Engineering.")
    st.stop()

X = engineered_df.drop(columns=[target_col]).select_dtypes(include=["number"])
y_raw = engineered_df[target_col]
y = LabelEncoder().fit_transform(y_raw.astype(str)) if (is_classification and not pd.api.types.is_numeric_dtype(y_raw)) else y_raw

_MODEL_REGISTRY = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=500),
    "Random Forest Classifier": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosted Trees": lambda: GradientBoostingClassifier(random_state=42),
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest Regressor": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosted Trees Regressor": lambda: GradientBoostingRegressor(random_state=42),
}
_SCORING = "roc_auc" if is_classification else "neg_root_mean_squared_error"
if is_classification and len(set(y)) > 2:
    _SCORING = "accuracy"  # roc_auc needs binarization for multiclass; keep this demo simple & robust

if st.button("▶️ Run Training", type="primary") or st.session_state.get("training_result"):
    if not st.session_state.get("training_result"):
        results = []
        progress = st.progress(0.0, text="Training models...")
        for i, model_spec in enumerate(plan.candidate_models):
            builder = _MODEL_REGISTRY.get(model_spec.name)
            if builder is None:
                continue
            model = builder()
            start = time.time()
            try:
                cv = min(5, max(2, len(X) // 20)) if len(X) >= 10 else 2
                scores = cross_val_score(model, X, y, cv=cv, scoring=_SCORING)
                score = float(scores.mean())
            except Exception as exc:  # noqa: BLE001
                score = float("nan")
                st.warning(f"{model_spec.name} failed to train: {exc}")
            elapsed = time.time() - start
            results.append({"name": model_spec.name, "score": score, "seconds": round(elapsed, 2)})
            progress.progress((i + 1) / len(plan.candidate_models), text=f"Trained {model_spec.name}")
        progress.empty()

        results = [r for r in results if r["score"] == r["score"]]  # drop NaNs
        results.sort(key=lambda r: r["score"], reverse=True)
        best = results[0] if results else None

        context = {
            "scoring_metric": _SCORING,
            "results": results,
            "best_model": best["name"] if best else None,
        }
        local_reasoning = AIReasoning(
            summary=(
                f"Trained {len(results)} candidate model(s) via cross-validation on {_SCORING}. "
                f"'{best['name']}' scored highest." if best else "No models trained successfully."
            ),
            evidence=[f"{r['name']}: {r['score']:.4f} ({_SCORING})" for r in results],
            confidence=0.78 if best else 0.3,
            alternatives_considered=[r["name"] for r in results[1:]],
            risks=["Cross-validation score may not generalize to a held-out production distribution."],
            expected_impact="The top-scoring model becomes the champion evaluated in the next stage.",
            intervention_recommended=not best or (len(results) > 1 and abs(results[0]["score"] - results[1]["score"]) < 0.01),
        )
        reasoning = generate_with_fallback(
            task_type="model_training_reasoning",
            subject="Model training results",
            context=context,
            fallback=local_reasoning,
        )
        st.session_state["training_result"] = {
            "results": results,
            "scoring": _SCORING,
            "best": best,
            "reasoning": reasoning,
            "context": context,
        }

    result = st.session_state["training_result"]
    results, scoring, best, reasoning = result["results"], result["scoring"], result["best"], result["reasoning"]

    if results:
        st.plotly_chart(
            plot_utils.model_comparison_bar([r["name"] for r in results], [r["score"] for r in results], scoring),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
        if best:
            st.success(f"🏆 Best model: **{best['name']}** — {scoring}: {best['score']:.4f}")
            st.session_state["selected_model"] = best["name"]
    else:
        st.error("No models completed training successfully.")

    ai_observation.render(reasoning, key="model_training", expanded=True)

    def _on_revise(feedback: str) -> None:
        updated = get_client().refine_reasoning(
            task_type="model_training_reasoning",
            subject="Model training results",
            context=st.session_state["training_result"]["context"],
            previous=st.session_state["training_result"]["reasoning"],
            feedback=feedback,
        )
        st.session_state["training_result"]["reasoning"] = updated

    ai_revision.render(key="model_training", on_revise=_on_revise, subject_label="the model training results")

    approval.render(
        stage_name="Model Training",
        approval_key="approved_training",
        next_step_label="Evaluation",
        override_note_key="training_override_note",
        on_approve=lambda: st.session_state.update(current_step=6),
    )
else:
    st.info("Click **Run Training** to fit and cross-validate the approved candidate models.")