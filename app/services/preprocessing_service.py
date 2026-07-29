"""Preprocessing planning service - turns per-feature AI observations from
the DatasetProfile into a concrete, orderable list of preprocessing steps.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from schemas.dataset_profile import DatasetProfile
from schemas.feature_profile import AIReasoning, FeatureRole, SemanticType
from services.langgraph_client import LangGraphUnavailable, get_client


class PreprocessingStep(BaseModel):
    step_id: str
    title: str
    applies_to: List[str]
    method: str
    reasoning: AIReasoning


class PreprocessingPlan(BaseModel):
    steps: List[PreprocessingStep]


class PreprocessingService:
    def __init__(self) -> None:
        self._client = get_client()

    def build_plan(self, profile: DatasetProfile) -> PreprocessingPlan:
        steps: List[PreprocessingStep] = []

        drop_cols = [
            f.name for f in profile.features.values()
            if f.role == FeatureRole.IDENTIFIER or f.leakage_risk
        ]
        if drop_cols:
            steps.append(self._step(
                "drop_columns", "Drop non-predictive columns", drop_cols,
                "Remove identifiers and leakage-risk columns",
                evidence=[f"{len(drop_cols)} column(s) flagged as identifier or leakage risk."],
                risks=["Verify none of these columns are actually needed downstream (e.g. as a join key)."],
            ))

        impute_cols = [f.name for f in profile.features.values() if 0 < f.missing_pct < 30 and f.role == FeatureRole.FEATURE]
        if impute_cols:
            steps.append(self._step(
                "impute_missing", "Impute missing values", impute_cols,
                "Median imputation (numeric) / mode imputation (categorical)",
                evidence=[f"{len(impute_cols)} column(s) have between 0-30% missing values."],
                risks=["Imputation can dilute signal if missingness itself is informative - consider adding indicator flags."],
            ))

        scale_cols = [
            f.name for f in profile.features.values()
            if f.semantic_type == SemanticType.NUMERICAL and f.role == FeatureRole.FEATURE
        ]
        if scale_cols:
            steps.append(self._step(
                "scale_numeric", "Scale numeric features", scale_cols,
                "StandardScaler",
                evidence=[f"{len(scale_cols)} numeric feature(s) detected requiring comparable scale."],
                risks=["Tree-based models are scale-invariant; scaling mainly benefits linear/distance-based models."],
            ))

        encode_cols = [
            f.name for f in profile.features.values()
            if f.semantic_type in (SemanticType.CATEGORICAL, SemanticType.ORDINAL, SemanticType.BINARY)
            and f.role == FeatureRole.FEATURE
        ]
        if encode_cols:
            steps.append(self._step(
                "encode_categorical", "Encode categorical features", encode_cols,
                "One-hot encoding (low cardinality) / frequency encoding (high cardinality)",
                evidence=[f"{len(encode_cols)} categorical/ordinal/binary feature(s) detected."],
                risks=["One-hot encoding high-cardinality columns can explode dimensionality."],
            ))

        skewed_cols = [
            f.name for f in profile.features.values()
            if f.skewness is not None and abs(f.skewness) >= 0.5 and f.role == FeatureRole.FEATURE
        ]
        if skewed_cols:
            steps.append(self._step(
                "transform_skew", "Correct skewed distributions", skewed_cols,
                "Log / Yeo-Johnson transform",
                evidence=[f"{len(skewed_cols)} feature(s) show |skewness| >= 0.5."],
                risks=["Transforms make coefficients harder to interpret in linear models."],
            ))

        return PreprocessingPlan(steps=steps)

    def refine_step_reasoning(self, step: PreprocessingStep, feedback: str) -> AIReasoning:
        """Re-invoke the agent with the data scientist's feedback on a
        single preprocessing step's reasoning."""
        context = {"step_id": step.step_id, "title": step.title, "applies_to": step.applies_to, "method": step.method}
        return self._client.refine_reasoning(
            task_type="preprocessing_step_reasoning",
            subject=step.title,
            context=context,
            previous=step.reasoning,
            feedback=feedback,
        )

    def _step(self, step_id, title, applies_to, method, evidence, risks) -> PreprocessingStep:
        context = {"step_id": step_id, "title": title, "applies_to": applies_to, "method": method, "evidence": evidence}
        try:
            reasoning = self._client.generate_reasoning(
                task_type="preprocessing_step_reasoning", subject=title, context=context,
            )
        except LangGraphUnavailable:
            reasoning = AIReasoning(
                summary=f"{title} recommended for {len(applies_to)} column(s) using {method}.",
                evidence=evidence,
                confidence=0.82,
                alternatives_considered=["Leave column(s) untouched", "Alternative encoding/scaling method"],
                risks=risks,
                expected_impact="Directly shapes the feature matrix handed to Feature Engineering / Model Training.",
                intervention_recommended=False,
            )
        return PreprocessingStep(step_id=step_id, title=title, applies_to=applies_to, method=method, reasoning=reasoning)
