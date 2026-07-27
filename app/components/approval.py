import streamlit as st


def render_dataset_approval():
	approved = st.checkbox("I have reviewed the dataset profile", value=st.session_state.approved_profile)
	st.session_state.approved_profile = approved
	if approved:
		st.success("Profile reviewed. Continue to Experiment Plan from the sidebar.")
