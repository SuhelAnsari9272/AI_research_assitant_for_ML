from pydantic import BaseModel
from typing import List
from enum import Enum

from schemas.dataset_profile import DatasetProfile

class ProblemSummary(BaseModel):

    project_goal: str
    problem_type: str
    target_column: str
    dataset_summary: DatasetProfile #str

class PreprocessingPlan(BaseModel):

    missing_value_strategy: list[str]
    encoding_strategy: list[str]
    scaling_required: bool
    feature_selection_required: bool
    remove_identifier_columns: bool

class ApprovalCheckpoint(BaseModel):

    approval_required: bool
    reason: str


class Metric(str, Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    PRECISION = "precision"
    RECALL = "recall"

class ValidationStrategy(str, Enum):
    TRAIN_TEST_SPLIT = "train_test_split"
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"

class EvaluationPlan(BaseModel):

    primary_metric: Metric
    secondary_metrics: list[Metric]
    validation_strategy: ValidationStrategy

class ModelName(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVM = "svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    MLP = "mlp"

class ModelRecommendation(BaseModel):

    model_name: ModelName
    reason: str
    priority: int  


class ExperimentPlan(BaseModel):

    preprocessing: PreprocessingPlan
    models: list[ModelRecommendation]
    evaluation: EvaluationPlan
    risks: list[str]
    # approval: ApprovalCheckpoint


# class PlannerOutput(BaseModel):

#     project_goal: str

#     problem_type: str

#     evaluation_metrics: List[str]

#     recommended_models: List[str]

#     preprocessing_steps: List[str]

#     train_strategy: str

#     warnings: List[str]

#     assumptions: List[str]

#     execution_plan: List[str]

