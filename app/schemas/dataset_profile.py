"""Dataset-level profile schema aggregating per-feature profiles."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from schemas.feature_profile import FeatureProfile


class DatasetProfile(BaseModel):
    """Aggregate statistics + per-column profiles for the uploaded dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_rows: int
    n_cols: int
    duplicate_rows: int
    missing_cells: int
    missing_pct: float
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    features: Dict[str, FeatureProfile] = Field(default_factory=dict)

