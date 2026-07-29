
from __future__ import annotations

from typing import List, Optional

import streamlit as st

from schemas.feature_profile import FeatureRole


def render(feature_names: List[str], selected: Optional[str], roles: Optional[dict] = None) -> str:
    """Render the pill navigator and return the currently selected feature name."""
    if selected not in feature_names:
        selected = feature_names[0]

    search = st.text_input("🔍 Jump to feature", placeholder="Type to filter features...", label_visibility="collapsed")
    visible = [f for f in feature_names if search.lower() in f.lower()] if search else feature_names

    with st.container(height=140, border=True):
        n_cols = 6
        rows = [visible[i : i + n_cols] for i in range(0, len(visible), n_cols)]
        for row in rows:
            cols = st.columns(n_cols)
            for c, name in zip(cols, row):
                role = roles.get(name) if roles else None
                icon = "🎯" if role == FeatureRole.TARGET else ("🔑" if role == FeatureRole.IDENTIFIER else "▫️")
                is_active = name == selected
                label = f"{icon} {name}"
                if c.button(label, key=f"pill_{name}", width="stretch", type="primary" if is_active else "secondary"):
                    selected = name

    return selected
