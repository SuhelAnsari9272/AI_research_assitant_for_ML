import os
import json
import pandas as pd

def load_input(problem_statement_path, data_path ):
    
    # problem_statement 
    with open(problem_statement_path, "r", encoding="utf-8") as f:
        problem_statement = json.load(f)

    # dataset involved
    dataset = pd.read_csv(data_path)

    return problem_statement, dataset


if __name__=="__main__":

    problem_statement_path = r"input\problem_statement.json"
    data_path = r"input\titanic.csv"

    problem_statement, data = load_input(problem_statement_path, data_path)

    # print(problem_statement)
    # print(type(problem_statement))
