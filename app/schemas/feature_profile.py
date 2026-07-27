
from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class SemanticType(str, Enum):
    """AI-inferred semantic meaning of a column (richer than the pandas dtype)."""

    NUMERICAL = "Numerical"
    CATEGORICAL = "Categorical"
    ORDINAL = "Ordinal"
    BINARY = "Binary"
    BOOLEAN = "Boolean"
    DATETIME = "Datetime"
    TEXT = "Text"
    IDENTIFIER = "Identifier"
    URL = "URL"
    EMAIL = "Email"
    PHONE = "Phone"
    GEO = "Geo"
    TARGET = "Target"
    UNKNOWN = "Unknown"


class FeatureRole(str, Enum):
    """The functional role a column plays in the modeling workflow."""

    IDENTIFIER = "Identifier"
    FEATURE = "Feature"
    TARGET = "Target"
    TIMESTAMP = "Timestamp"


class AIReasoning(BaseModel):
    """Transparency payload rendered inside the 'AI Reasoning' expander.

    Every AI decision surfaced in the UI (feature semantics, preprocessing,
    model choice, etc.) should be explainable through this same structure so
    the data scientist always sees *why*, not just *what*.
    """

    summary: str = Field(..., description="One or two sentence plain-English rationale.")
    evidence: List[str] = Field(default_factory=list, description="Concrete statistics/signals used.")
    alternatives_considered: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    expected_impact: str = Field(..., description="What happens if this recommendation is accepted.")
    intervention_recommended: bool = Field(
        default=False, description="True if the AI itself thinks a human should double check this."
    )


class TopCategory(BaseModel):
    value: str
    count: int
    pct: float


class FeatureProfile(BaseModel):
    """Full statistical + semantic profile for a single dataframe column."""

    name: str
    pandas_dtype: str
    semantic_type: SemanticType
    suggested_dtype: str
    role: FeatureRole
    leakage_risk: bool = False

    unique_values: int
    cardinality_pct: float
    missing_count: int
    missing_pct: float

    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outlier_count: Optional[int] = None

    memory_usage_bytes: int

    sample_values: List[str] = Field(default_factory=list)
    top_categories: List[TopCategory] = Field(default_factory=list)

    ai_observations: List[str] = Field(default_factory=list)
    recommended_transformations: List[str] = Field(default_factory=list)
    potential_risks: List[str] = Field(default_factory=list)

    reasoning: AIReasoning
