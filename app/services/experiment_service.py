"""Experiment planning service - proposes candidate models, validation
strategy and the primary metric, each with an `AIReasoning` payload.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from schemas.dataset_profile import DatasetProfile
from schemas.feature_profile import AIReasoning
from services.langgraph_client import LangGraphUnavailable, get_client


class CandidateModel(BaseModel):
    name: str
    library: str
    rationale: str
    expected_strength: str


class ExperimentPlan(BaseModel):
    problem_type: str
    primary_metric: str
    validation_strategy: str
    candidate_models: List[CandidateModel]
    reasoning: AIReasoning


_CLASSIFICATION_POOL = [
    CandidateModel(
        name="Logistic Regression",
        library="scikit-learn",
        rationale="Strong, fast baseline; highly interpretable coefficients.",
        expected_strength="Baseline / interpretability",
    ),
    CandidateModel(
        name="Random Forest Classifier",
        library="scikit-learn",
        rationale="Handles non-linearities and mixed feature types with minimal tuning.",
        expected_strength="Robust general-purpose accuracy",
    ),
    CandidateModel(
        name="Gradient Boosted Trees",
        library="scikit-learn",
        rationale="Typically the strongest tabular performer; captures complex interactions.",
        expected_strength="Highest accuracy ceiling",
    ),
]

_REGRESSION_POOL = [
    CandidateModel(
        name="Linear Regression",
        library="scikit-learn",
        rationale="Fast, interpretable baseline for continuous targets.",
        expected_strength="Baseline / interpretability",
    ),
    CandidateModel(
        name="Random Forest Regressor",
        library="scikit-learn",
        rationale="Captures non-linear relationships without heavy feature engineering.",
        expected_strength="Robust general-purpose accuracy",
    ),
    CandidateModel(
        name="Gradient Boosted Trees Regressor",
        library="scikit-learn",
        rationale="Strong performance on structured/tabular numeric data.",
        expected_strength="Highest accuracy ceiling",
    ),
]


class ExperimentService:
    """Builds an `ExperimentPlan` from a `DatasetProfile`."""

    def __init__(self) -> None:
        self._client = get_client()

    def build_plan(self, profile: DatasetProfile, evaluation_metric_hint: str = "") -> ExperimentPlan:
        is_classification = (profile.problem_type or "").lower().startswith("class")
        pool = _CLASSIFICATION_POOL if is_classification else _REGRESSION_POOL

        n_rows = profile.n_rows
        validation_strategy = (
            "Stratified 5-fold cross-validation" if is_classification else "5-fold cross-validation"
        )
        if n_rows < 500:
            validation_strategy = "Leave-more-out CV (small dataset: <500 rows) with repeated resampling"

        primary_metric = evaluation_metric_hint or (
            "ROC-AUC" if is_classification else "RMSE"
        )

        evidence = [
            f"Dataset has {n_rows:,} rows and {profile.n_cols} columns.",
            f"Problem type set to '{profile.problem_type}'.",
        ]
        if n_rows < 500:
            evidence.append("Row count is low, so simpler, higher-bias-tolerant validation was preferred.")

        context = {
            "n_rows": n_rows,
            "n_cols": profile.n_cols,
            "problem_type": profile.problem_type,
            "candidate_models": [m.name for m in pool],
            "primary_metric": primary_metric,
            "validation_strategy": validation_strategy,
            "evidence": evidence,
        }
        reasoning = self._generate_reasoning(context, pool, validation_strategy)

        return ExperimentPlan(
            problem_type=profile.problem_type or "Unknown",
            primary_metric=primary_metric,
            validation_strategy=validation_strategy,
            candidate_models=pool,
            reasoning=reasoning,
        )

    def refine_plan_reasoning(self, plan: ExperimentPlan, profile: DatasetProfile, feedback: str) -> AIReasoning:
        """Re-invoke the agent with the data scientist's feedback on the
        current experiment plan's reasoning."""
        context = {
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "problem_type": profile.problem_type,
            "candidate_models": [m.name for m in plan.candidate_models],
            "primary_metric": plan.primary_metric,
            "validation_strategy": plan.validation_strategy,
        }
        return self._client.refine_reasoning(
            task_type="experiment_plan_reasoning",
            subject=f"Experiment plan for {profile.problem_type}",
            context=context,
            previous=plan.reasoning,
            feedback=feedback,
        )

    def _generate_reasoning(self, context: dict, pool, validation_strategy) -> AIReasoning:
        try:
            return self._client.generate_reasoning(
                task_type="experiment_plan_reasoning",
                subject=f"Experiment plan for {context['problem_type']}",
                context=context,
            )
        except LangGraphUnavailable:
            pass

        return AIReasoning(
            summary=(
                f"Proposed {len(pool)} candidate models spanning a linear baseline, a bagging "
                f"ensemble, and a boosting ensemble, evaluated via {validation_strategy}."
            ),
            evidence=context["evidence"],
            confidence=0.8,
            alternatives_considered=["Single-model quick-fit", "AutoML exhaustive search across 20+ algorithms"],
            risks=["Boosted trees can overfit on very small datasets without careful regularization."],
            expected_impact="Determines which models are actually trained in the Model Training stage.",
            intervention_recommended=False,
        )