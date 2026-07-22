
import pandas as pd
import json
from graph import create_workflow_graph
from utils import save_graph
from groq import BadRequestError
import json

try:

    graph = create_workflow_graph()
    assitant = graph.compile()
    save_graph(assitant)


    problem_statement_path = "sample_problem_statement.json"
    result = assitant.invoke({'problem_statement_path' : problem_statement_path })
    # print(experiment_plan)

except BadRequestError as e:
    print(json.dumps(e.body, indent=4))
    raise


print(result["experiment_plan"])




# # print(experiment_plan)
# print("********************************************")
# print(experiment_plan.problem_summary)

# print("------------------------------")

# print(experiment_plan.preprocessing)

# print("------------------------------")

# print(experiment_plan.models)

# print("------------------------------")

# print(experiment_plan.evaluation)

# print("------------------------------")

# print(experiment_plan.execution_steps)

# print("------------------------------")

# print(experiment_plan.risks)

# print("------------------------------")

# print(experiment_plan.approval)