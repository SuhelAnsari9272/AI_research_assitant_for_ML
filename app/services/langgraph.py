from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.graph import create_workflow_graph
from groq import BadRequestError
from schemas.dataset_profile import DatasetProfile
from schemas.experiment import ExperimentPlan
import pandas as pd


def execute_langgraph_workflow(project_config, dataset: pd.DataFrame) -> tuple[DatasetProfile, ExperimentPlan]:
    graph = create_workflow_graph()
    assistant = graph.compile()
    try:
        result = assistant.invoke({"project_config": project_config, "dataset": dataset})
    except BadRequestError as error:
        raise RuntimeError(error.body) from error

    dataset_profile = result.get("dataset_profile")
    experiment_plan = result.get("experiment_plan")
    if dataset_profile is None or experiment_plan is None:
        raise RuntimeError("langGraph workflow returned incomplete results")
    return dataset_profile, experiment_plan
