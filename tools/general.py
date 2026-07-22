import json
from schemas.project_config import ProjectConfig

def load_input(problem_statement_path: str, ) -> ProjectConfig :

    with open(problem_statement_path, 'r') as f:
        problem_statement = json.load(f)

    return ProjectConfig(
        project_name=problem_statement['project_name'],
        objective= problem_statement['objective'],
        target_column = problem_statement['target_column'],
        problem_type = problem_statement["problem_type"],
        preferred_metric=problem_statement['preferred_metric'],
        constraints= problem_statement['constraints'],
        dataset_filename=problem_statement['dataset_filename']
    )

