from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Streamlit page scripts are launched with app/ as the import root.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

from schemas.dataset_profile import (
	ClassificationTargetProfile,
	ColumnInfo,
	DatasetProfile,
	GeneralProfile,
	HighCardinalityColumn,
	MissingValueInfo,
	NumericalColumnStatistics,
	QualityProfile,
	RegressionTargetProfile,
	StatisticsProfile,
)


def _safe_float(value: object) -> float | None:
	if value is None or pd.isna(value) or not np.isfinite(float(value)):
		return None
	return float(value)


def get_dataset_profile(df: pd.DataFrame, target_column: str | None = None) -> DatasetProfile:
	"""Build a lightweight, reproducible profile from the uploaded dataframe."""

    
	columns_by_dtype: dict[str, list[ColumnInfo]] = {}
	for column in df.columns:
		dtype_name = str(df[column].dtype)
		columns_by_dtype.setdefault(dtype_name, []).append(
			ColumnInfo(
				column_name=str(column),
				sample_values=[str(value) for value in df[column].dropna().unique()[:5]],
			)
		)

	numerical_statistics: list[NumericalColumnStatistics] = []
	for column in df.select_dtypes(include=np.number).columns:
		series = df[column].dropna()
		numerical_statistics.append(
			NumericalColumnStatistics(
				column_name=str(column), mean=_safe_float(series.mean()), median=_safe_float(series.median()),
				std=_safe_float(series.std()), minimum=_safe_float(series.min()), maximum=_safe_float(series.max()),
				q1=_safe_float(series.quantile(0.25)), q3=_safe_float(series.quantile(0.75)),
				skewness=_safe_float(series.skew()), kurtosis=_safe_float(series.kurtosis()),
			)
		)

	missing_columns = [
		MissingValueInfo(column_name=str(column), missing_count=int(df[column].isna().sum()),
						 missing_percentage=round(float(df[column].isna().mean() * 100), 2))
		for column in df.columns if df[column].isna().any()
	]
	unique_identifier_columns = [
		str(column) for column in df.columns if len(df) and df[column].nunique(dropna=False) == len(df)
	]
	high_cardinality_columns = [
		HighCardinalityColumn(column_name=str(column), unique_count=int(df[column].nunique(dropna=False)),
							  unique_percentage=round(float(df[column].nunique(dropna=False) / len(df) * 100), 2))
		for column in df.columns if len(df) and df[column].nunique(dropna=False) / len(df) >= 0.5
	]
	quality = QualityProfile(
		duplicate_rows=int(df.duplicated().sum()),
		duplicate_percentage=round(float(df.duplicated().mean() * 100), 2) if len(df) else 0.0,
		missing_columns=missing_columns,
		constant_columns=[str(column) for column in df.columns if df[column].nunique(dropna=False) <= 1],
		unique_identifier_columns=unique_identifier_columns,
		high_cardinality_columns=high_cardinality_columns,
		mostly_empty_columns=[str(column) for column in df.columns if df[column].isna().mean() >= 0.5],
		warnings=[f"{item.column_name} contains {item.missing_percentage:.2f}% missing values." for item in missing_columns],
	)

	target = None
	if target_column and target_column in df.columns:
		series = df[target_column].dropna()
		if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
			target = RegressionTargetProfile(
				target_column=target_column, problem_type="Regression", mean=float(series.mean()),
				median=float(series.median()), std=float(series.std()), minimum=float(series.min()),
				maximum=float(series.max()), skewness=float(series.skew()),
			)
		else:
			counts = series.astype(str).value_counts()
			target = ClassificationTargetProfile(
				target_column=target_column, problem_type="Classification",
				classes=[str(value) for value in counts.index],
				class_distribution={str(key): int(value) for key, value in counts.items()},
				imbalance_ratio=float(counts.max() / counts.min()) if len(counts) else 0.0,
				majority_class=str(counts.index[0]) if len(counts) else "",
				minority_class=str(counts.index[-1]) if len(counts) else "",
			)

	return DatasetProfile(
		general=GeneralProfile(n_rows=len(df), n_cols=len(df.columns), columns_by_dtype=columns_by_dtype),
		statistics=StatisticsProfile(numerical_statistics=numerical_statistics), quality=quality, target=target,
	)
