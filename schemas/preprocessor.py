from pydantic import BaseModel

class PreprocessingResult(BaseModel):

    transformed_columns: list[str]

    removed_columns: list[str]

    generated_features: list[str]

    preprocessing_steps: list[str]

    train_shape: tuple[int, int]

    test_shape: tuple[int, int]

    warnings: list[str]