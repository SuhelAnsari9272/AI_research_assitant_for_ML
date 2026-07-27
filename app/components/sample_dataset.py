from __future__ import annotations

import pandas as pd
import streamlit as st


def render(df: pd.DataFrame, n: int = 10) -> None:
    """Render a tabbed preview of the dataframe (head/tail/random sample)."""
    tab_head, tab_tail, tab_random = st.tabs(["Head", "Tail", "Random Sample"])
    with tab_head:
        st.dataframe(df.head(n), width="stretch")
    with tab_tail:
        st.dataframe(df.tail(n), width="stretch")
    with tab_random:
        sample_n = min(n, len(df))
        st.dataframe(df.sample(sample_n, random_state=42) if sample_n else df, width="stretch")