from pydantic import BaseModel
from enum import Enum

class ProblemType(str, Enum) :
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class ProjectConfig(BaseModel):
    project_name: str
    objective: str
    target_column: str
    problem_type: str
    preferred_metric: str #str | None = None
    # constraints: dict = {}
    # dataset_filename :str

