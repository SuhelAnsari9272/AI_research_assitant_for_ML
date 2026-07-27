
import pandas as pd
import json
from src.graph import create_workflow_graph
from utils import save_graph
from groq import BadRequestError
import json
from utils import to_jsonable

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


# print(result["experiment_plan"])
# print(result)

### see results 

plan = result["experiment_plan"].model_dump()

with open("experiment_plan.json", "w") as f:
    json.dump(plan, f, indent=4)

data_profile = result["dataset_profile"].model_dump()

with open("data_profile.json", "w") as f:
    json.dump(data_profile, f, indent=4)
