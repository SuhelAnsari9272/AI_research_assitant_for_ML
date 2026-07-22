
from state import State
from prompts import PLANNER_SYSTEM_PROMPT
import pandas as pd
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import llm
from schemas.experiment import ExperimentPlan

from tools.general import load_input
from tools.pandas_tools import analyse_dataset


def input_loader(state :State):

    problem_statement_path = state['problem_statement_path']
    project_config = load_input(problem_statement_path)

    return {"project_config" : project_config}

def input_dataset(state:State):

    dataset_filename = state['project_config'].dataset_filename
    dataset = pd.read_csv(os.path.join("input", dataset_filename))

    return {"dataset" : dataset}


def data_profiler(state:State):
    
    project_config = state['project_config']
    dataset = state['dataset']

    dataset_profile = analyse_dataset(dataset, project_config)

    return {"dataset_profile": dataset_profile}


def planner(state:State):

    dataset_profile = state['dataset_profile']
    project_config = state['project_config']

    planner_user_prompt = f"""
            ## Project Configuration

            {project_config.model_dump_json(indent=2)}

            ------------------------------------------------

            ## Dataset Profile

            {dataset_profile.model_dump_json(indent=2)}

            ------------------------------------------------

            Create the experiment plan.
            """

    experiment_plan = llm.with_structured_output(ExperimentPlan).invoke([
                SystemMessage(content = PLANNER_SYSTEM_PROMPT),
                HumanMessage(content = planner_user_prompt)
                ])


    return {'experiment_plan' : experiment_plan }

