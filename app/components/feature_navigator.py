import streamlit as st


def render_feature_navigator(columns_by_dtype):
	options = [
		info.column_name
		for infos in columns_by_dtype.values()
		for info in infos
	]
	return st.selectbox("Inspect a column", options)
