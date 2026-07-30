from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import re
import warnings
from scipy import stats as scipy_stats

from schemas.dataset_profile import DatasetProfile


from schemas.feature_profile import (
    AIReasoning,
    FeatureProfile,
    FeatureRole,
    SemanticType,
    TopCategory,
)

from services.langgraph_client import get_client


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
_PHONE_RE = re.compile(r"^\+?[\d\-\(\) ]{7,15}$")
# _ID_NAME_RE = re.compile(r"(^id$|id$|^id_|_id$|uuid|guid)", re.IGNORECASE)
_ID_NAME_RE = re.compile(r"(^id$|(^|[_-])id([_-]|$)|id$|uuid|guid)", re.IGNORECASE,)


class DatasetProfilerService:
	"""Encapsulates all dataset + feature profiling logic."""

	def __init__(self) -> None :
		self._client = get_client()

	def profile_dataset(self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None, ) -> DatasetProfile:

		n_rows, n_cols = df.shape
		missing_cells = int(df.isna().sum().sum())
		total_cells = max(n_rows * n_cols, 1)

		features = {
            col: self._profile_feature(df, col, target_column, problem_type)
            for col in df.columns
        }

		return DatasetProfile(
			n_rows=n_rows,
            n_cols=n_cols,
            duplicate_rows=int(df.duplicated().sum()),
            missing_cells=missing_cells,
            missing_pct=round(100 * missing_cells / total_cells, 2),
            target_column=target_column,
            problem_type=problem_type,
            features=features,
        )


	# Per-feature profiling
    # ------------------------------------------------------------------ #
	def _profile_feature(
        self,
        df: pd.DataFrame,
        col: str,
        target_column: Optional[str],
        problem_type: Optional[str],) -> FeatureProfile:

		series = df[col]
		n_rows = len(series)
		missing_count = int(series.isna().sum())
		missing_pct = round(100 * missing_count / max(n_rows, 1), 2)
		unique_values = int(series.nunique(dropna=True))
		cardinality_pct = round(100 * unique_values / max(n_rows, 1), 2)

		semantic_type,  type_evidence = self._infer_semantic_type(series, col)
		role = self._infer_role(col, target_column, semantic_type)

		is_numeric = (semantic_type in (SemanticType.INTEGER, SemanticType.FLOAT, SemanticType.BINARY, ) and pd.api.types.is_numeric_dtype(series)
)

		stats_dict = self._numeric_stats(series) if is_numeric else {}
		outlier_count = self._count_outliers(series) if is_numeric else None
		top_categories = self._top_categories(series) if semantic_type in (
            SemanticType.CATEGORICAL, SemanticType.ORDINAL, SemanticType.BINARY, SemanticType.BOOLEAN ) else []

		leakage_risk = self._check_leakage_risk(col, target_column, unique_values, n_rows)

		observations, transformations, risks = self._build_observations(
            series=series,
            semantic_type=semantic_type,
            role=role,
            missing_pct=missing_pct,
            cardinality_pct=cardinality_pct,
            unique_values=unique_values,
            stats_dict=stats_dict,
            outlier_count=outlier_count,
            leakage_risk=leakage_risk,
        	)

		reasoning = self._build_local_reasoning(
            col=col,
            semantic_type=semantic_type,
            type_evidence=type_evidence,
            observations=observations,
            risks=risks,
        	)

		return FeatureProfile(
            name=col,
            pandas_dtype=str(series.dtype),
            semantic_type=semantic_type,
            suggested_dtype=self._suggest_dtype(semantic_type, series),
            role=role,
            leakage_risk=leakage_risk,

            unique_values=unique_values,
            cardinality_pct=cardinality_pct,
            missing_count=missing_count,
            missing_pct=missing_pct,
            # memory_usage_bytes=int(series.memory_usage(deep=True)),

            sample_values=[str(v) for v in series.dropna().head(5).tolist()],
            top_categories=top_categories,

            ai_observations=observations,
            recommended_transformations=transformations,
            potential_risks=risks,

            reasoning=reasoning,

            **stats_dict,
            outlier_count=outlier_count,
        )

	# Semantic type inference
    # ------------------------------------------------------------------ #

	def _infer_semantic_type(self, series: pd.Series, col: str):
		"""Returns (SemanticType, evidence[list[str]])."""
		n = len(series)
		non_null = series.dropna()
		unique_ratio = non_null.nunique() / max(len(non_null), 1)
		evidence: List[str] = []

		if pd.api.types.is_datetime64_any_dtype(series):
			evidence.append("Column dtype is already datetime64.")
			return SemanticType.DATETIME, evidence

		if non_null.empty:
			evidence.append("Column is entirely empty.")
			return SemanticType.UNKNOWN, evidence
		
		sample_str = non_null.astype(str).head(50)

		if _ID_NAME_RE.search(col) or unique_ratio > 0.9:  # and
			evidence.append(f"Column name matches identifier pattern and {unique_ratio:.0%} of values are unique.")
			return SemanticType.IDENTIFIER, evidence

		if sample_str.map(lambda v: bool(_EMAIL_RE.match(v))).mean() > 0.8:
			evidence.append("Over 80% of sampled values match an email pattern.")
			return SemanticType.EMAIL, evidence

		if sample_str.map(lambda v: bool(_URL_RE.match(v))).mean() > 0.8:
			evidence.append("Over 80% of sampled values match a URL pattern.")
			return SemanticType.URL, evidence

		if col.lower() in {"lat", "latitude", "lon", "lng", "longitude"} or col.lower() in {"geo", "location"}:
			evidence.append("Column name suggests geographic coordinates.")
			return SemanticType.GEO, evidence

		if not pd.api.types.is_numeric_dtype(series):
			try:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					parsed = pd.to_datetime(non_null.head(50), errors="coerce")
					if parsed.notna().mean() > 0.85:
						evidence.append("Over 85% of sampled values parse successfully as dates.")
						return SemanticType.DATETIME, evidence
					
			except Exception:
				pass

		if sample_str.map(lambda v: bool(_PHONE_RE.match(v))).mean() > 0.8 and not pd.api.types.is_numeric_dtype(series):
			evidence.append("Over 80% of sampled values match a phone-number pattern.")
			return SemanticType.PHONE, evidence

		if pd.api.types.is_bool_dtype(series):
			evidence.append("Column dtype is boolean.")
			return SemanticType.BOOLEAN, evidence

		if unique_ratio <= 1.0 and non_null.nunique() == 2:
			evidence.append("Exactly two distinct values found.")
			return SemanticType.BINARY, evidence


		if pd.api.types.is_numeric_dtype(series):
			if unique_ratio > 0.95 and _ID_NAME_RE.search(col):  
				evidence.append("High uniqueness with an identifier-like name.")
				return SemanticType.IDENTIFIER, evidence
			evidence.append(f"Numeric dtype with {non_null.nunique()} distinct values ({unique_ratio:.0%} unique).")

			if non_null.nunique() <= 15 and unique_ratio < 0.05:
				evidence.append("Low cardinality relative to row count suggests an ordinal/categorical code.")
				return SemanticType.ORDINAL,  evidence

			return SemanticType.FLOAT, evidence

		avg_len = sample_str.str.len().mean()
		if non_null.nunique() > 50 and avg_len > 30:
			evidence.append(f"High cardinality ({non_null.nunique()} values) with long average text length ({avg_len:.0f} chars).")
			return SemanticType.TEXT, evidence

		evidence.append(f"Object dtype with {non_null.nunique()} distinct values ({unique_ratio:.0%} unique).")
		return SemanticType.CATEGORICAL, evidence

	def _infer_role(self, col: str, target_column: Optional[str], semantic_type: SemanticType) -> FeatureRole:
		if target_column and col == target_column:
			return FeatureRole.TARGET
		if semantic_type == SemanticType.IDENTIFIER:
			return FeatureRole.IDENTIFIER
		if semantic_type == SemanticType.DATETIME:
			return FeatureRole.TIMESTAMP
		return FeatureRole.FEATURE

	def _suggest_dtype(self, semantic_type: SemanticType, series: pd.Series) -> str:

		mapping = {
			SemanticType.INTEGER : 'int64',
			SemanticType.FLOAT : 'float64',
			SemanticType.CATEGORICAL: "category",
			SemanticType.ORDINAL: "category",
			SemanticType.BINARY: "category",
			SemanticType.BOOLEAN: "bool",
			SemanticType.DATETIME: "datetime64[ns]",
			SemanticType.TEXT: "string",
			SemanticType.IDENTIFIER: "string",
			SemanticType.URL: "string",
			SemanticType.EMAIL: "string",
			SemanticType.PHONE: "string",
			SemanticType.GEO: "float64",
			SemanticType.TARGET: str(series.dtype),
			SemanticType.UNKNOWN: str(series.dtype),
		}
		return mapping.get(semantic_type, str(series.dtype))

	# Statistics
    # ------------------------------------------------------------------ #
	def _numeric_stats(self, series: pd.Series) -> dict:
		numeric = pd.to_numeric(series, errors="coerce").dropna()
		if numeric.empty:
			return {}
		return {
			"min": float(numeric.min()),
			"max": float(numeric.max()),
			"mean": float(numeric.mean()),
			"median": float(numeric.median()),
			"std": float(numeric.std()) if len(numeric) > 1 else 0.0,
			"variance": float(numeric.var()) if len(numeric) > 1 else 0.0,
			"skewness": float(scipy_stats.skew(numeric)) if len(numeric) > 2 else 0.0,
			"kurtosis": float(scipy_stats.kurtosis(numeric)) if len(numeric) > 3 else 0.0,
		}

	def _count_outliers(self, series: pd.Series) -> int:
		numeric = pd.to_numeric(series, errors="coerce").dropna()
		if len(numeric) < 4:
			return 0
		q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
		iqr = q3 - q1
		if iqr == 0:
			return 0
		lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
		return int(((numeric < lower) | (numeric > upper)).sum())

	def _top_categories(self, series: pd.Series, top_n: int = 8) -> List[TopCategory]:
		n = len(series)
		counts = series.value_counts(dropna=True).head(top_n)
		return [
			TopCategory(value=str(idx), count=int(cnt), pct=round(100 * cnt / max(n, 1), 2))
			for idx, cnt in counts.items()
        ]

	def _check_leakage_risk(self, col: str, target_column: Optional[str], unique_values: int, n_rows: int) -> bool:
		if not target_column or col == target_column:
			return False
		suspicious_terms = ("target", "label", "outcome", "result", "score_final")
		return any(term in col.lower() for term in suspicious_terms)

	# Observations + reasoning
    # ------------------------------------------------------------------ #
	def _build_observations(
		self,
		series: pd.Series,
		semantic_type: SemanticType,
		role: FeatureRole,
		missing_pct: float,
		cardinality_pct: float,
		unique_values: int,
		stats_dict: dict,
		outlier_count: Optional[int],
		leakage_risk: bool,
		):
		observations: List[str] = []
		transformations: List[str] = []
		risks: List[str] = []

		if role == FeatureRole.IDENTIFIER:
			observations.append("Identifier detected - unique per row, holds no predictive signal.")
			transformations.append("Exclude from model features.")

		if leakage_risk:
			observations.append("Potential target leakage - column name resembles the target/outcome.")
			risks.append("May leak information only known after prediction time.")

		if missing_pct == 0:
			observations.append("No missing values.")
		elif missing_pct < 5:
			observations.append("Missing values negligible.")
		elif missing_pct < 30:
			observations.append(f"Moderate missingness ({missing_pct:.1f}%).")
			transformations.append("Impute missing values (median/mode) or add a missing-indicator flag.")
		else:
			observations.append(f"High missingness ({missing_pct:.1f}%).")
			risks.append("Column may be unreliable or worth dropping if missingness exceeds business tolerance.")
			transformations.append("Consider dropping column or advanced imputation.")

		if semantic_type == SemanticType.CATEGORICAL and cardinality_pct > 40:
			observations.append(f"High cardinality categorical ({unique_values} unique values).")
			transformations.append("Use target/frequency encoding instead of one-hot to avoid dimensionality blow-up.")
		elif semantic_type in (SemanticType.CATEGORICAL, SemanticType.BINARY, SemanticType.ORDINAL):
			transformations.append("One-hot / ordinal encode for modeling.")

		skew = stats_dict.get("skewness")
		if skew is not None:
			if abs(skew) < 0.5:
				observations.append("Distribution nearly normal.")
			elif skew >= 0.5:
				observations.append(f"Strong positive skew (skewness={skew:.2f}).")
				transformations.append("Apply log or Box-Cox transformation.")
			else:
				observations.append(f"Strong negative skew (skewness={skew:.2f}).")
				transformations.append("Apply power/Yeo-Johnson transformation.")

		if outlier_count:
			pct = round(100 * outlier_count / max(len(series), 1), 1)
			if outlier_count > 0:
				observations.append(f"{outlier_count} outliers detected via IQR rule ({pct}% of rows).")
				transformations.append("Consider robust scaling or winsorization.")

		if semantic_type == SemanticType.FLOAT:
			transformations.append("Standard/MinMax scaling recommended before distance-based or linear models.")

		return observations, transformations, risks

	def _build_local_reasoning(
		self,
		col: str,
		semantic_type: SemanticType,
		type_evidence: List[str],
		observations: List[str],
		risks: List[str],
		) -> AIReasoning:
		"""Fast, deterministic reasoning used to populate the profile
		instantly for every column. Bulk profiling never calls the LLM
		(hundreds of columns x one Groq call each would be slow and
		expensive) - `generate_llm_reasoning()` below upgrades this
		heuristic explanation to a live Groq/LangGraph one, lazily, only
		for the feature(s) the data scientist actually opens on Page 2.
		"""
		alternatives = self._alternative_types(semantic_type)
		return AIReasoning(
			summary=(
				f"Classified '{col}' as {semantic_type.value} based on dtype, cardinality, "
				f"and value-pattern analysis."
			),
			evidence=type_evidence + observations[:3],
			alternatives_considered=alternatives,
			risks=risks if risks else ["No significant risks detected for this classification."],
			expected_impact=(
				"Accepting this type drives which preprocessing transforms are recommended "
				"in the next stage (encoding vs. scaling vs. exclusion)."
			),
			intervention_recommended=bool(risks) #type_confidence < 0.65 or bool(risks),
		)

	# On-demand LLM reasoning (lazy — called from Page 2, one feature at a time)
    # ------------------------------------------------------------------ #
	def generate_llm_reasoning(self, feature: FeatureProfile) -> AIReasoning:
		"""Ask the Groq/LangGraph agent to reason about this single feature,
		grounded in its already-computed statistics. Raises
		`LangGraphUnavailable` if no Groq key is configured or the call fails
		- callers should catch this and keep showing the local heuristic.
		"""
		context = self._feature_context(feature)
		return self._client.generate_reasoning(
			task_type="feature_semantic_reasoning",
			subject=feature.name,
			context=context,
		)

	def refine_llm_reasoning(self, feature: FeatureProfile, feedback: str) -> AIReasoning:
		"""Re-invoke the agent with the data scientist's feedback on the
		feature's current reasoning (whichever - heuristic or LLM - is
		currently displayed)."""
		context = self._feature_context(feature)
		return self._client.refine_reasoning(
			task_type="feature_semantic_reasoning",
			subject=feature.name,
			context=context,
			previous=feature.reasoning,
			feedback=feedback,
		)

	def _feature_context(self, feature: FeatureProfile) -> dict:
		"""Serialize exactly the evidence the LLM is allowed to reason from."""
		return {
			"column_name": feature.name,
			"pandas_dtype": feature.pandas_dtype,
			"current_semantic_type": feature.semantic_type.value,
			"current_role": feature.role.value,
			"unique_values": feature.unique_values,
			"cardinality_pct": feature.cardinality_pct,
			"missing_pct": feature.missing_pct,
			"min": feature.min, "max": feature.max, "mean": feature.mean,
			"median": feature.median, "std": feature.std,
			"skewness": feature.skewness, "kurtosis": feature.kurtosis,
			"outlier_count": feature.outlier_count,
			"sample_values": feature.sample_values,
			"top_categories": [c.value for c in feature.top_categories],
			"existing_ai_observations": feature.ai_observations,
			"leakage_risk_flagged": feature.leakage_risk,
		}

	def _alternative_types(self, chosen: SemanticType) -> List[str]:
		pool = [t.value for t in SemanticType if t != chosen and t != SemanticType.TARGET]
		# Keep the two semantically closest-looking alternatives for a concise panel.
		neighbors = {
			SemanticType.INTEGER: [SemanticType.ORDINAL, SemanticType.IDENTIFIER],
			SemanticType.CATEGORICAL: [SemanticType.ORDINAL, SemanticType.TEXT],
			SemanticType.ORDINAL: [SemanticType.CATEGORICAL, SemanticType.INTEGER],
			SemanticType.IDENTIFIER: [SemanticType.INTEGER, SemanticType.CATEGORICAL],
			SemanticType.DATETIME: [SemanticType.TEXT, SemanticType.CATEGORICAL],
		}.get(chosen, [SemanticType.CATEGORICAL, SemanticType.INTEGER])
		return [n.value for n in neighbors if n.value in pool]