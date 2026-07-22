from schemas.experiment import ProblemSummary
from schemas.experiment import ExperimentPlan

from tools.project_config import get_project_config
from tools.pandas_tools import analyse_dataset

def get_problem_summary(dataset, problem_statement_path) -> ProblemSummary :

    project_config = get_project_config(problem_statement_path)
    dataset_summary = analyse_dataset(dataset, project_config)

    return ProblemSummary(
        project_goal=project_config.objective,
        problem_type=project_config.problem_type,
        target_column=project_config.target_column,
        dataset_summary=dataset_summary,
    )


def get_preprocessing_plan() : 
    pass  ## Use LLM for the preprocessing plan

def get_model_recommendation() :
    pass ## use LLM for model recommendation

def get_evaluation_plan():
    pass

def experiment_human_approval():
    pass


def get_experiment_plan(dataset, problem_statement_path) :

    return ExperimentPlan(
        problem_summary= get_problem_summary(dataset, problem_statement_path), 
        preprocessing= get_preprocessing_plan(),
        models = get_model_recommendation(),
        evaluation = get_evaluation_plan(),
        execution_steps = execution_steps,
        risks = risks,
        approval = experiment_human_approval()
    )