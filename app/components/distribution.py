"""Renders the correct distribution chart for a feature based on its
AI-inferred semantic type."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from schemas.feature_profile import FeatureProfile, SemanticType
from utils import plot_utils


def render(series: pd.Series, feature: FeatureProfile) -> None:
    """Render the appropriate Plotly distribution chart for this feature."""
    st.markdown("##### Distribution")

    if (feature.semantic_type == SemanticType.INTEGER) | (feature.semantic_type == SemanticType.FLOAT):
        fig = plot_utils.numeric_distribution(series, title=f"{feature.name} — distribution")
    elif feature.semantic_type == SemanticType.DATETIME:
        fig = plot_utils.datetime_distribution(series, title=f"{feature.name} — records over time")
    elif feature.semantic_type in (
        SemanticType.CATEGORICAL, SemanticType.ORDINAL, SemanticType.BINARY, SemanticType.BOOLEAN,
    ):
        fig = plot_utils.categorical_distribution(series, title=f"{feature.name} — top categories")
    else:
        st.info(f"No distribution chart available for semantic type '{feature.semantic_type.value}'.")
        return

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
