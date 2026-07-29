"""Renders the full statistical + semantic detail panel for one feature."""
from __future__ import annotations

import streamlit as st

from schemas.feature_profile import FeatureProfile
from utils.formatting import (
    format_number,
    format_pct,
    truncate,
)


def _badge(text: str, kind: str = "gray") -> str:
    return f'<span class="aa-badge aa-badge-{kind}">{text}</span>'


def render(feature: FeatureProfile) -> None:
    """Render name/type header, the full stat grid, samples, observations."""

    header_badges = " ".join([
        _badge(feature.semantic_type.value, "blue"),
        _badge(feature.role.value, "gray"),
    ])

    if feature.leakage_risk:
        header_badges += " " + _badge("⚠ Leakage Risk", "red")

    st.markdown(f"### {feature.name}")
    st.markdown(header_badges, unsafe_allow_html=True)
    st.write("")

    # --- Type + dtype row -------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Detected dtype**  \n`{feature.pandas_dtype}`")
    c2.markdown(f"**Suggested dtype**  \n`{feature.suggested_dtype}`")
    c3.markdown(f"**Role**  \n{feature.role.value}")

    st.divider()

    # --- Core stat grid -----------------------------------------------------
    st.markdown("##### Statistics")
    grid_specs = [
        ("Unique Values", format_number(feature.unique_values)),
        ("Cardinality", format_pct(feature.cardinality_pct)),
        ("Missing Values", format_number(feature.missing_count)),
        ("Missing %", format_pct(feature.missing_pct)),
        ("Min", format_number(feature.min)),
        ("Max", format_number(feature.max)),
        ("Mean", format_number(feature.mean)),
        ("Median", format_number(feature.median)),
        ("Std Dev", format_number(feature.std)),
        ("Variance", format_number(feature.variance)),
        ("Skewness", format_number(feature.skewness)),
        ("Kurtosis", format_number(feature.kurtosis)),
        ("Outlier Count", format_number(feature.outlier_count)),
        # ("Memory Usage", format_bytes(feature.memory_usage_bytes)),
    ]
    cols = st.columns(4)
    for i, (label, value) in enumerate(grid_specs):
        with cols[i % 4]:
            st.markdown(
                f'<div class="aa-card" style="margin-bottom:0.6rem;">'
                f'<div class="aa-metric-label">{label}</div>'
                f'<div class="aa-metric-value" style="font-size:1.05rem;">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # --- Samples / top categories -------------------------------------------
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("##### Sample Values")
        if feature.sample_values:
            for v in feature.sample_values:
                st.code(truncate(v, 60), language=None)
        else:
            st.caption("No non-null sample values available.")

    with sc2:
        st.markdown("##### Top Categories")
        if feature.top_categories:
            for cat in feature.top_categories:
                st.markdown(f"`{truncate(cat.value, 24)}` — {cat.count:,} ({cat.pct:.1f}%)")
        else:
            st.caption("Not applicable for this semantic type.")

    st.divider()

    # --- Observations / transforms / risks ---------------------------------
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown("##### 🔎 AI Observations")
        for obs in feature.ai_observations or ["No notable observations."]:
            st.markdown(f"- {obs}")
    with oc2:
        st.markdown("##### 🛠 Recommended Transformations")
        for t in feature.recommended_transformations or ["None required."]:
            st.markdown(f"- {t}")
    with oc3:
        st.markdown("##### ⚠️ Potential Risks")
        for r in feature.potential_risks or ["No significant risks identified."]:
            st.markdown(f"- {r}")
