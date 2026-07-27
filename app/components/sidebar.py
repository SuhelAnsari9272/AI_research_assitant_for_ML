import streamlit as st


def render_sidebar(current_page: str = ""):
	with st.sidebar:
		st.markdown("## AutoML workspace")
		config = st.session_state.get("project_config")
		if config:
			if hasattr(config, "model_dump"):
				config = config.model_dump()
			st.caption(config.get("project_name") or "Untitled project")
			st.write(f"**{config.get('problem_type', 'Problem')}**")
		st.divider()
		st.page_link("Home.py", label="Workspace overview")
		st.page_link("pages/1_Project.py", label="1. Project & dataset")
		st.page_link("pages/2_Dataset_Profile.py", label="2. Dataset profile")
		st.page_link("pages/3_Experiment_Plan.py", label="3. Experiment plan")
		if current_page:
			st.caption(f"Current: {current_page}")
