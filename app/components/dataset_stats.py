import streamlit as st
import pandas as pd

from schemas.dataset_profile import StatisticsProfile


def render_statistics(statistics: StatisticsProfile):
    st.subheader("📈 Numerical Statistics")

    if not statistics.numerical_statistics:
        st.info("No numerical columns found.")
        return

    rows = []

    for stat in statistics.numerical_statistics:
        rows.append({
            "Column": stat.column_name,
            "Mean": stat.mean,
            "Median": stat.median,
            "Std": stat.std,
            "Min": stat.minimum,
            "Q1": stat.q1,
            "Q3": stat.q3,
            "Max": stat.maximum,
            "Skewness": stat.skewness,
            "Kurtosis": stat.kurtosis,
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

def render_statistics_details(statistics: StatisticsProfile):
    st.subheader("📊 Detailed Statistics")

    for stat in statistics.numerical_statistics:
        with st.expander(f"📌 {stat.column_name}", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                st.metric("Mean", f"{stat.mean:.3f}" if stat.mean is not None else "-")
                st.metric("Median", f"{stat.median:.3f}" if stat.median is not None else "-")
                st.metric("Std", f"{stat.std:.3f}" if stat.std is not None else "-")
                st.metric("Minimum", f"{stat.minimum:.3f}" if stat.minimum is not None else "-")
                st.metric("Maximum", f"{stat.maximum:.3f}" if stat.maximum is not None else "-")

            with c2:
                st.metric("Q1", f"{stat.q1:.3f}" if stat.q1 is not None else "-")
                st.metric("Q3", f"{stat.q3:.3f}" if stat.q3 is not None else "-")
                st.metric("Skewness", f"{stat.skewness:.3f}" if stat.skewness is not None else "-")
                st.metric("Kurtosis", f"{stat.kurtosis:.3f}" if stat.kurtosis is not None else "-")


def render_statistics_dashboard(statistics: StatisticsProfile):
    st.subheader("📈 Numerical Statistics")
    df = pd.DataFrame(
		[s.model_dump() for s in statistics.numerical_statistics]
	).rename(columns={
		"column_name": "Column",
		"minimum": "Min",
		"maximum": "Max",
		"skewness": "Skew",
		"kurtosis": "Kurtosis",
	})
    st.dataframe(
		df.style.format("{:.3f}", na_rep="-"),
		use_container_width=True,
		hide_index=True,
	)