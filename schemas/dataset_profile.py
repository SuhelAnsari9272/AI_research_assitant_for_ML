from pydantic import BaseModel, Field
from typing import List, Optional

class NumericalColumnStatistics(BaseModel):
    column_name: str

    mean: Optional[float]
    median: Optional[float]

    std: Optional[float]

    minimum: Optional[float]
    maximum: Optional[float]

    q1: Optional[float]
    q3: Optional[float]

    skewness: Optional[float]
    kurtosis: Optional[float]

class StatisticsProfile(BaseModel):
    numerical_statistics : list[NumericalColumnStatistics]


class HighCardinalityColumn(BaseModel):

    column_name: str
    unique_count: int
    unique_percentage: float


class MissingValueInfo(BaseModel):

    column_name: str
    missing_count: int
    missing_percentage: float


class QualityProfile(BaseModel):

    duplicate_rows: int
    duplicate_percentage: float

    missing_columns: list[MissingValueInfo]
    constant_columns: list[str]

    unique_identifier_columns: list[str]
    high_cardinality_columns: list[HighCardinalityColumn]

    mostly_empty_columns: list[str]
    warnings: list[str]


class GeneralProfile(BaseModel):
    n_rows :int
    n_cols : int
    numerical_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]

class BaseTargetProfile(BaseModel):

    target_column: str
    problem_type: str

class ClassificationTargetProfile(BaseTargetProfile):

    classes: list[str]
    class_distribution: dict
    imbalance_ratio: float
    majority_class: str
    minority_class: str

class RegressionTargetProfile(BaseTargetProfile):

    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    skewness: float


class DatasetProfile(BaseModel):
    general: GeneralProfile
    statistics: StatisticsProfile
    quality: QualityProfile
    target: BaseTargetProfile | None