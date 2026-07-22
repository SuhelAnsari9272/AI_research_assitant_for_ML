
import json
from schemas. dataset_profile import BaseTargetProfile
from schemas.project_config import ProjectConfig

def get_project_config(problem_statement_path: str) -> ProjectConfig :

    with open(problem_statement_path, 'r') as f:
        problem_statement = json.load(f)

    return ProjectConfig(
        project_name=problem_statement['project_name'],
        objective= problem_statement['objective'],
        target_column = problem_statement['target_column'],
        problem_type = problem_statement["problem_type"],
        preferred_metric=problem_statement['preferred_metric'],
        constraints= problem_statement['constraints']
    )

    
def get_base_target_profile(project_config : ProjectConfig) -> BaseTargetProfile :

    return BaseTargetProfile(
        target_column=project_config.target_column,
        problem_type= project_config.problem_type
    )


if __name__=="__main__":

    problem_statement_path = r"sample_problem_statement.json"
    project_configuration = get_project_config(problem_statement_path)

    print(project_configuration)
