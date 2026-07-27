from __future__ import annotations

import pandas as pd
import numpy as np

DTYPE_OPTIONS = [
	"keep",
	"string",
	"category",
	"Int64",
	"float64",
	"boolean",
	"datetime64[ns]",
]


def convert_column_dtype(series: pd.Series, dtype_name: str) -> pd.Series:
	if dtype_name == "keep":
		return series
	
	if dtype_name == "Int64":
		try:
			numeric = pd.to_numeric(series, errors="raise")
		except (TypeError, ValueError) as error:
			raise ValueError("values must be numeric whole numbers") from error
		non_null = numeric.dropna()
		if not np.allclose(non_null, np.floor(non_null)):
			raise ValueError("values must be whole numbers; use float64 for fractional values")
		return numeric.astype("Int64")
	
	if dtype_name == "float64":
		try:
			return pd.to_numeric(series, errors="raise").astype("float64")
		except (TypeError, ValueError) as error:
			raise ValueError("values must be numeric") from error
		
	if dtype_name == "boolean":
		if pd.api.types.is_bool_dtype(series):
			return series.astype("boolean")
		
		values = series.astype("string").str.strip().str.lower()
		mapped = values.map(
			{
				"true": True, 
				"false": False, 
				"1": True, 
				"0": False,
			})
		invalid = series.notna() & mapped.isna()
		if invalid.any():
			raise ValueError("values must be true/false or 1/0")
		return mapped.astype("boolean")

	if dtype_name == "datetime64[ns]":
		return pd.to_datetime(series, errors="raise")
	
	return series.astype(dtype_name)


def apply_dtype_overrides(
	df: pd.DataFrame, overrides: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, str]]:
	updated = df.copy()
	applied: dict[str, str] = {}
	for column, dtype_name in overrides.items():
		if column not in updated.columns or dtype_name == "keep":
			continue
		updated[column] = convert_column_dtype(updated[column], dtype_name)
		applied[column] = dtype_name
	return updated, applied
