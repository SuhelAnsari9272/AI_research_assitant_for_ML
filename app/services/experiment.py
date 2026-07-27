from __future__ import annotations

import json
from pathlib import Path


def get_experiment_plan() -> dict | None:
	artifact = Path(__file__).resolve().parents[2] / "artifacts" / "experiment_plan.json"
	if not artifact.exists():
		return None
	with artifact.open(encoding="utf-8") as handle:
		return json.load(handle)
