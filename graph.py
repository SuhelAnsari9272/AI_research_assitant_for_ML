from langgraph.graph import StateGraph, START, END
from state import State
from nodes import planner, input_loader, input_dataset, data_profiler

def create_workflow_graph()  :

    graph = StateGraph(State)
    graph.add_node("input_loader", input_loader)
    graph.add_node("input_dataset", input_dataset)
    graph.add_node("data_profiler", data_profiler)
    graph.add_node("planner", planner)


    graph.add_edge(START, "input_loader")
    graph.add_edge("input_loader", "input_dataset")
    graph.add_edge("input_dataset", "data_profiler")
    graph.add_edge("data_profiler", "planner")
    return graph