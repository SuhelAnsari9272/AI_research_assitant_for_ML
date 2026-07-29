"""Page 7 — Model Evaluation.

Refits the champion model on a train/test split and reports holdout metrics
plus a confusion matrix (classification) or predicted-vs-actual (regression).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from components import ai_observation, ai_revision, approval
from components.sidebar import render as render_sidebar
from schemas.feature_profile import AIReasoning
from schemas.state import init_session_state
from services.langgraph_client import generate_with_fallback, get_client
from utils import plot_utils

st.set_page_config(page_title="Model Evaluation", page_icon="🧬", layout="wide")
init_session_state()
render_sidebar()

st.title("7️⃣ Model Evaluation")
st.caption("Holdout evaluation of the champion model selected in Model Training.")

profile = st.session_state.get("dataset_profile")
plan = st.session_state.get("experiment_plan")
fe_plan = st.session_state.get("feature_engineering_plan")
training_result = st.session_state.get("training_result")

if not all([profile, plan, fe_plan, training_result]) or not st.session_state.get("approved_training"):
    st.warning("Approve Model Training (Page 6) before evaluating.")
    st.stop()

st.session_state["current_step"] = 6

is_classification = plan.problem_type.lower().startswith("class")
target_col = profile.target_column
engineered_df: pd.DataFrame = fe_plan["engineered_df"]
best_name = training_result["best"]["name"] if training_result["best"] else None

if not best_name:
    st.error("No champion model available from training.")
    st.stop()

_MODEL_REGISTRY = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=500),
    "Random Forest Classifier": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosted Trees": lambda: GradientBoostingClassifier(random_state=42),
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest Regressor": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosted Trees Regressor": lambda: GradientBoostingRegressor(random_state=42),
}

X = engineered_df.drop(columns=[target_col]).select_dtypes(include=["number"])
y_raw = engineered_df[target_col]
label_encoder = None
if is_classification and not pd.api.types.is_numeric_dtype(y_raw):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
else:
    y = y_raw

if st.session_state.get("evaluation_result") is None:
    with st.spinner(f"Refitting {best_name} on a held-out split..."):
        stratify = y if is_classification and len(set(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
        model = _MODEL_REGISTRY[best_name]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics, evidence = {}, []
        if is_classification:
            metrics["Accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            metrics["F1 (weighted)"] = round(f1_score(y_test, y_pred, average="weighted"), 4)
            if len(set(y_test)) == 2 and hasattr(model, "predict_proba"):
                metrics["ROC-AUC"] = round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(set(y_test) | set(y_pred))
            evidence.append(f"Holdout size: {len(y_test)} rows (20% split).")
        else:
            metrics["RMSE"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
            metrics["MAE"] = round(mean_absolute_error(y_test, y_pred), 4)
            metrics["R²"] = round(r2_score(y_test, y_pred), 4)
            cm, labels = None, None
            evidence.append(f"Holdout size: {len(y_test)} rows (20% split).")

        context = {
            "champion_model": best_name,
            "problem_type": plan.problem_type,
            "metrics": metrics,
            "holdout_fraction": 0.2,
        }
        local_reasoning = AIReasoning(
            summary=f"Evaluated '{best_name}' on a held-out 20% split using metrics appropriate for {plan.problem_type.lower()}.",
            evidence=evidence + [f"{k}: {v}" for k, v in metrics.items()],
            confidence=0.8,
            alternatives_considered=["Nested cross-validation for tighter confidence intervals"],
            risks=["Single holdout split has higher variance than repeated CV; consider re-running with a different seed."],
            expected_impact="These metrics are what get reported at Deployment sign-off.",
            intervention_recommended=False,
        )
        reasoning = generate_with_fallback(
            task_type="evaluation_reasoning",
            subject=f"Evaluation of {best_name}",
            context=context,
            fallback=local_reasoning,
        )

        st.session_state["evaluation_result"] = {
            "metrics": metrics,
            "y_test": y_test,
            "y_pred": y_pred,
            "confusion_matrix": cm,
            "labels": labels,
            "reasoning": reasoning,
            "context": context,
        }

eval_result = st.session_state["evaluation_result"]

st.markdown(f"##### Champion model: `{best_name}`")
metric_cols = st.columns(len(eval_result["metrics"]))
for col, (name, value) in zip(metric_cols, eval_result["metrics"].items()):
    col.metric(name, value)

if is_classification and eval_result["confusion_matrix"] is not None:
    st.plotly_chart(
        plot_utils.confusion_matrix_heatmap(eval_result["confusion_matrix"], eval_result["labels"]),
        width="stretch", config={"displayModeBar": False},
    )
elif not is_classification:
    st.plotly_chart(
        plot_utils.residuals_scatter(pd.Series(eval_result["y_test"]), pd.Series(eval_result["y_pred"])),
        width="stretch", config={"displayModeBar": False},
    )

ai_observation.render(eval_result["reasoning"], key="evaluation", expanded=True)


def _on_revise(feedback: str) -> None:
    updated = get_client().refine_reasoning(
        task_type="evaluation_reasoning",
        subject=f"Evaluation of {best_name}",
        context=st.session_state["evaluation_result"]["context"],
        previous=st.session_state["evaluation_result"]["reasoning"],
        feedback=feedback,
    )
    st.session_state["evaluation_result"]["reasoning"] = updated


ai_revision.render(key="evaluation", on_revise=_on_revise, subject_label="the evaluation results")

approval.render(
    stage_name="Model Evaluation",
    approval_key="approved_evaluation",
    next_step_label="Deployment",
    override_note_key="evaluation_override_note",
    on_approve=lambda: st.session_state.update(current_step=7),
)
