import streamlit as st


def render_feature_card(column_profile):
	st.subheader(column_profile.column_name)
	st.write("Sample values")
	st.code(", ".join(column_profile.sample_values) or "No non-null samples")
