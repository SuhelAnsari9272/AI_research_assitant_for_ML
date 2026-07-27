import streamlit as st


def render_dataset_summary(df, general, quality):
	st.subheader("Dataset health")
	cards = st.columns(4)
	cards[0].metric("Rows", f"{general.n_rows:,}")
	cards[1].metric("Columns", general.n_cols)
	cards[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
	cards[3].metric("Duplicate rows", f"{quality.duplicate_rows:,}")

	if quality.warnings:
		# st.warning(f"Found {len(quality.warnings)} warnings")
		st.markdown("### ⚠️ Warnings")


		for warning in quality.warnings :
			 
			if isinstance(warning, list):

				if warning:
					message = warning[0]  # First line as title
					if len(warning) > 1:
						message += "\n\n" + "\n".join(f"- {item}" for item in warning[1:])
					st.warning(message)
			else  :
				st.warning(warning)

	else:
		st.success("No missing-value warnings detected.")
