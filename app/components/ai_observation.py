import streamlit as st
from schemas.human_review import HumanReview
import pandas as pd

def render_ai_observation(column_profile):
	st.subheader("Observation")
	st.info(
		f"{column_profile.column_name} has {len(column_profile.sample_values)} representative "
		"non-null sample values in this profile. Review its type and business meaning before training."
	)


def human_review_ui(df:pd.DataFrame) -> HumanReview:
	columns = list(df.columns)

	st.subheader("📝 Human Review")
	st.caption("Optional: Override automatic profiling only where necessary.")

	# Target Override

	target_override = None

	if st.checkbox("Override target column"):
		target_override = st.selectbox("Target Column",options=columns,)

	# Ignore Columns
	ignored_columns = []
	if st.checkbox("Ignore specific columns"):
		ignored_columns = st.multiselect("Columns to ignore", options=columns,)

	# protected columns
	protected_columns = []
	if st.checkbox("Protected / Sensitive columns"):
		protected_columns = st.multiselect("Protected columns", options=columns, )

	# dtype overrides
	dtype_overrides = {}

	if st.checkbox("Override data types"):
		n = st.number_input("Number of overrides", min_value=1, max_value=len(columns), value=1, step=1,)

		dtype_options = [
				"Int64",
				"float64",
				"string",
				"boolean",
				"datetime64[ns]",
				"category",
			]
		
		used_columns  = set()

		for i in range(n):
			c1, c2 = st.columns([2, 2])

			with c1:
				available = [c for c in columns if c not in used_columns ]
				if not available:
					break

				column = st.selectbox(f"Column {i+1}", available, key=f"dtype_col_{i}",)
				used_columns.add(column)

			with c2:
				dtype = st.selectbox("New dtype", dtype_options, key=f"dtype_value_{i}",)
				dtype_overrides[column] = dtype

	## feature roles
	
	feature_roles = {}

	if st.checkbox("Override feature roles"):
		n = st.number_input("Number of role overrides", min_value=1, max_value=len(columns), value=1, step=1,)
		role_options = [
            "identifier",
            "feature",
            "target",
            "timestamp",
            "categorical",
            "numerical",
            "text",
        	]

		used_columns = set()

		for i in range(n):
			c1, c2 = st.columns([2, 2])

			with c1:
				available = [c for c in columns if c not in used_columns]
				if not available:
					break

				column = st.selectbox(f"Role Column {i+1}", available, key=f"role_col_{i}", )
				used_columns.add(column)

			with c2:
				role = st.selectbox("Role", role_options, key=f"role_value_{i}", )

			feature_roles[column] = role


	# Business contrainsts 
	constraints = []
	if st.checkbox("Add business constraints"):
		txt = st.text_area(
            "One constraint per line",
            placeholder=
			"""
			Examples:
			Age cannot be negative
			Salary must be positive
			Customer_ID must remain unique
			""", 
			height=150, )

		constraints = [line.strip() for line in txt.splitlines() if line.strip()]


	# Notes
	notes = ""
	if st.checkbox("Additional notes"):
		notes = st.text_area("Notes", height=120, )

	return HumanReview(
		dtype_overrides=dtype_overrides,
		ignored_columns=ignored_columns,
		target_override=target_override,
		protected_columns=protected_columns,
		feature_roles=feature_roles,
		business_constraints=constraints,
		notes = notes
	)