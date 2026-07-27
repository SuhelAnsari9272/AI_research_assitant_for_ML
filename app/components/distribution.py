import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def render_distribution(df, column_profile):
	column = column_profile.column_name
	series = df[column].dropna()
	st.subheader("Distribution")
	if series.empty:
		st.info("No data available.")
		return

	    # Numeric columns
	if pd.api.types.is_numeric_dtype(series):
		unique_count = series.nunique()

        # Heuristic: <=20 unique values => discrete
		if unique_count <= 20:
			counts = (
                series.value_counts()
                .sort_index()
                .rename_axis(column)
                .reset_index(name="Count")
            )
			st.bar_chart(
                counts.set_index(column)["Count"],
                use_container_width=True,
            )

		 # Continuous variable
		else:
			fig, ax = plt.subplots(figsize=(8, 4))
			ax.hist(series, bins="auto")
			ax.set_xlabel(column)
			ax.set_ylabel("Frequency")
			ax.set_title(f"{column} Distribution")
			st.pyplot(fig)

	# Categorical columns
	else:
		counts = (
            series.astype(str)
            .value_counts()
            .head(20)
        )
		st.bar_chart(counts, use_container_width=True)
