from typing import TypedDict

from schemas.dataset_profile import DatasetProfile
from schemas.experiment import ExperimentPlan
from schemas.project_config import ProjectConfig
import pandas as pd

class State(TypedDict):

    problem_statement_path : str
    project_config : ProjectConfig
    dataset : pd.DataFrame

    dataset_profile: DatasetProfile

    experiment_plan : ExperimentPlan

    # eda_output: dict

    # training_results: dict

    # comparison_result: dict

    # report: str


class AutoMLState(TypedDict):

    project_config: ProjectConfig

    dataset: pd.DataFrame

    dataset_profile: DatasetProfile

    experiment_plan: ExperimentPlan

    eda_report: dict

    training_results: dict

    comparison_result: dict

    final_report: str


# project_config

# dataset

# dataset_profile

# experiment_plan

# eda_result

# trained_models

# comparison_result

# report

# human_feedback

# execution_history

# current_step