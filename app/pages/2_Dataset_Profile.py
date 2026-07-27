import streamlit as st
import json
from utils import to_jsonable
from app.state import initialize

from components.sidebar import render_sidebar

from components.dataset_summary import render_dataset_summary

from app.components.dataset_stats import render_statistics

from components.feature_navigator import render_feature_navigator

from components.feature_card import render_feature_card

from components.distribution import render_distribution

from components.ai_observation import render_ai_observation, human_review_ui

from components.approval import render_dataset_approval
from components.target_distibution import render_target

from services.profiler import get_dataset_profile


from tools.pandas_tools import analyse_dataset


# --------------------------------------------------------

# Initialization

# --------------------------------------------------------

initialize()

st.set_page_config(

    page_title="Dataset Profile",
    layout="wide"

)



# --------------------------------------------------------

# Dataset Check

# --------------------------------------------------------

if st.session_state.dataset is None:
    st.error("No dataset uploaded.")
    st.stop()



df = st.session_state.dataset
project_config =  st.session_state.project_config


# --------------------------------------------------------

# Generate Profile Once

# --------------------------------------------------------

target_column = project_config.target_column #(st.session_state.project_config or {}).get("target_column")
print("target_column is : ", target_column )

if st.session_state.dataset_profile is None:

    with st.spinner("Profiling Dataset..."):

        st.session_state.dataset_profile = analyse_dataset(df,project_config) #get_dataset_profile(df, target_column)



profile = st.session_state.dataset_profile



# --------------------------------------------------------

# Sidebar

# --------------------------------------------------------

render_sidebar(current_page="Dataset Profile")



# --------------------------------------------------------
# Page Title
# --------------------------------------------------------

st.title("📊 Dataset Profile")
st.caption("Review the detected schema and approve before moving to Experiment Planning." )

# --------------------------------------------------------
# Dataset Summary
# --------------------------------------------------------


render_dataset_summary(df, profile.general, profile.quality)
st.divider()


# --------------------------------------------------------
# Dataset Preview
# --------------------------------------------------------

# render_sample_dataset(df)
render_statistics(profile.statistics)
st.divider()

# --------------------------------------------------------

# Feature Explorer

# --------------------------------------------------------

st.subheader("🔍 Feature Explorer")
selected_column = render_feature_navigator(profile.general.columns_by_dtype)



column_profile = next(
    info
    for infos in profile.general.columns_by_dtype.values()
    for info in infos
    if info.column_name == selected_column
)
st.divider()


# --------------------------------------------------------
# Feature Details
# --------------------------------------------------------

left, right = st.columns([2,2])


with left:
    render_feature_card(column_profile)


with right:
    render_distribution(df, column_profile)

st.divider()


# --------------------------------
# target Profile 
target_profile = profile.target

if not target_profile is None:
    render_target(target_profile)


# --------------------------------------------------------

# AI Observations

# --------------------------------------------------------

# render_ai_observation(column_profile)
human_review = human_review_ui(df)
st.divider()



# --------------------------------------------------------

# Approve Dataset

# --------------------------------------------------------

render_dataset_approval()

with open("human_review.json", "w") as f:
    json.dump(to_jsonable(human_review), f, indent=4)
# --------------------------------------------------------

# Debug

# --------------------------------------------------------

# st.write(profile)