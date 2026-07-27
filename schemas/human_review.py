from pydantic import BaseModel

class HumanReview(BaseModel):
    dtype_overrides: dict[str, str] = {}
    ignored_columns: list[str] = []
    target_override: str | None = None
    protected_columns: list[str] = []
    feature_roles: dict[str, str] = {}
    business_constraints: list[str] = []
    notes: str = ""
