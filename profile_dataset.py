import pandas as pd

from tools.project_config import get_project_config, get_base_target_profile
from tools.pandas_tools import analyse_dataset

problem_statement_path = r"sample_problem_statement.json"
dataset_path = r"input\titanic.csv"

project_config = get_project_config(problem_statement_path)

base_target_profile = get_base_target_profile(project_config)


dataset = pd.read_csv(dataset_path)
dataset_analysis = analyse_dataset(dataset, base_target_profile)

print(dataset_analysis)