import streamlit as st

DEFAULT_STATE = {

    "project_config": None,

    "dataset": None,

    "original_dataset": None,

    "dataset_dtype_overrides": {},

    "dataset_signature": None,

    "dataset_profile": None,

    "experiment_plan": None,

    "preprocessing_plan": None,

    "approved_profile": False,

    "approved_experiment": False,

    "approved_preprocessing": False
}


def initialize():

    for k, v in DEFAULT_STATE.items():

        if k not in st.session_state:

            st.session_state[k] = v