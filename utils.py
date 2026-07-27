from pydantic import BaseModel
from enum import Enum
import pandas as pd
import numpy as np


def save_graph(graph) : 

    mermaid_code = graph.get_graph().draw_mermaid()

    # print(mermaid_code)
    with open("artifacts/graph.mmd", "w") as f:
        f.write(mermaid_code)


    png_data = graph.get_graph().draw_mermaid_png()

    with open("artifacts/architecture.png", "wb") as f:
        f.write(png_data)

def to_jsonable(obj):
    # Pydantic models
    if isinstance(obj, BaseModel):
        return {k: to_jsonable(v) for k, v in obj.model_dump().items()}

    # DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    # Series
    if isinstance(obj, pd.Series):
        return obj.to_dict()

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # NumPy scalars
    if isinstance(obj, np.generic):
        return obj.item()

    # Enum
    if isinstance(obj, Enum):
        return obj.value

    # dict
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    return obj