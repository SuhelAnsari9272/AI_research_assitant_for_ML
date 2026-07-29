"""Top-of-page summary metric cards for the Dataset Profile page."""
from __future__ import annotations

import streamlit as st

from schemas.dataset_profile import DatasetProfile
from utils.formatting import format_bytes, format_number, format_pct


def render(profile: DatasetProfile) -> None:
    """Render the row of summary metric cards above the Feature Explorer."""
    cols = st.columns(6)
    metrics = [
        ("Rows", format_number(profile.n_rows)),
        ("Columns", format_number(profile.n_cols)),
        ("Duplicates", format_number(profile.duplicate_rows)),
        ("Missing Cells", f"{format_number(profile.missing_cells)} ({format_pct(profile.missing_pct)})"),
        ("Target", profile.target_column or "—"),
        ("Problem Type", profile.problem_type or "—"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""<div class="aa-card">
                    <div class="aa-metric-label">{label}</div>
                    <div class="aa-metric-value" style="font-size:1.15rem;">{value}</div>
                </div>""",
                unsafe_allow_html=True,
            )
