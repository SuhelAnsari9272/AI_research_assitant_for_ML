from __future__ import annotations

import pandas as pd
import streamlit as st
from code_editor import code_editor

from utils.pandas_sandbox import run_pandas_snippet
_EXAMPLE_SNIPPETS = [
    "df[df['age'] > 30]",
    "df.query(\"age > 30 and salary > 50000\")",
    "df.sort_values('salary', ascending=False).head(20)",
    "df.groupby('region')['salary'].mean().reset_index()",
    "df[df.isna().any(axis=1)]   # rows with any missing value",
]


_RUN_BUTTON = [{
    "name": "Run",
    "feather": "Play",
    "primary": True,
    "hasText": True,
    "showWithIcon": True,
    "commands": ["submit"],
    "style": {"bottom": "0.44rem", "right": "0.4rem"},
}]


def render(df: pd.DataFrame, n: int = 10, key_prefix: str = "sample_dataset") -> None:
    """Render a tabbed preview of the dataframe with a custom filtering tab."""
    tab_head, tab_tail, tab_random, tab_filter = st.tabs(["Head", "Tail", "Random Sample", "Filter by Code"])

    with tab_head:
        st.dataframe(df.head(n), width="stretch")

    with tab_tail:
        st.dataframe(df.tail(n), width="stretch")

    with tab_random:
        sample_n = min(n, len(df))
        st.dataframe(df.sample(sample_n, random_state=42) if sample_n else df, width="stretch")

    with tab_filter:
        _render_custom_filter(df, key_prefix=key_prefix)

def _render_custom_filter(df: pd.DataFrame, key_prefix: str) -> None:
    """A pandas code box: write an expression against `df`, run it, see the result."""
    st.caption(
        "Write pandas code against `df` (columns, `pd`, and `np` are all available). "
        "The last expression's value is shown - like a Jupyter cell - or set a "
        "`result = ...` variable yourself."
    )

    with st.expander("💡 Example snippets"):
        for snippet in _EXAMPLE_SNIPPETS:
            st.code(snippet, language="python")

    code_key = f"{key_prefix}_filter_code"
    result_key = f"{key_prefix}_filter_result"
    last_event_key = f"{key_prefix}_filter_last_event_id"
    default_code = st.session_state.get(code_key, "df.head(10)")\

    code_to_run: str
    run_now: bool

    
    response = code_editor(
        default_code,
        lang="python",
        theme="default",
        shortcuts="vscode",
        height=[6, 10],
        buttons=_RUN_BUTTON,
        key=f"{key_prefix}_code_editor",
    )
    # Keep the editor's latest content around so the box doesn't reset to
    # the original default if the tab/page is revisited later.
    if response["text"]:
        st.session_state[code_key] = response["text"]

    code_to_run = response["text"] or default_code
    run_now = response["type"] == "submit" and response["id"] != st.session_state.get(last_event_key)
    if run_now:
        st.session_state[last_event_key] = response["id"]

    clear_clicked = st.button("🗑️ Clear result", key=f"{key_prefix}_filter_clear")
    if clear_clicked:
        st.session_state.pop(result_key, None)

    if run_now:
        st.session_state[result_key] = run_pandas_snippet(code_to_run, df)

    result = st.session_state.get(result_key)

    if result is None:
        return

    if not result.ok:
        st.error(result.error)
        return

    value = result.value
    if isinstance(value, pd.Series):
        value = value.to_frame()

    if isinstance(value, pd.DataFrame):
        st.success(f"Result: {value.shape[0]:,} rows × {value.shape[1]:,} columns")
        st.dataframe(value, width="stretch")
        st.download_button(
            "⬇️ Download result as CSV",
            data=value.to_csv(index=True).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv",
            key=f"{key_prefix}_filter_download",
        )
    else:
        st.success("Result:")
        st.write(value)