import streamlit as st
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum

class ProblemType(str, Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"


class ProjectConfig(BaseModel):
    project_name: str = ""
    problem_type: ProblemType = ProblemType.CLASSIFICATION
    target_column: str = ""
    evaluation_metric: str = ""
    business_goal: str = ""


WORKFLOW_STEPS: List[str] = [
    "Project",
    "Dataset Profile",
    "Experiment Plan",
    "Preprocessing",
    "Feature Engineering",
    "Model Training",
    "Evaluation",
    "Deployment",
]


# Default value for every key the app touches. `None`/empty defaults keep
# pages easy to guard with `if not st.session_state.dataset: ...`.
_DEFAULTS: Dict[str, Any] = {
    "project_config": None,          # ProjectConfig
    "dataset": None,                 # pd.DataFrame
    "dataset_name": None,            # str
    "dataset_profile": None,         # DatasetProfile
    "experiment_plan": None,         # ExperimentPlan
    "preprocessing_plan": None,      # PreprocessingPlan
    "feature_engineering_plan": None,
    "training_result": None,
    "evaluation_result": None,
    "deployment_result": None,
    "deployment_reasoning": None,
    "feature_llm_reasoning": {},   # {feature_name: AIReasoning} cache of upgraded per-feature LLM reasoning
    # approval flags, one per workflow stage
    "approved_project": False,
    "approved_profile": False,
    "approved_experiment_plan": False,
    "approved_preprocessing": False,
    "approved_feature_engineering": False,
    "approved_training": False,
    "approved_evaluation": False,
    "approved_deployment": False,
    # navigation / selection state
    "current_step": 0,
    "selected_feature": None,
    "selected_preprocessing_step": None,
    "selected_model": None,
}

def init_session_state() -> None:
    """Populate any missing session-state keys with their defaults.

    Safe to call on every page render - existing values are never overwritten.
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

def reset_session_state() -> None:
    """Wipe the entire workflow back to a fresh project (used by Home)."""
    for key in list(_DEFAULTS.keys()):
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()