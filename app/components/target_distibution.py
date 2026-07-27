import streamlit as st
import pandas as pd

def render_target(target_profile):

    st.subheader(f"🎯 Target Analysis: {target_profile.target_column}")

    if target_profile.problem_type == "Classification":
        class_dist = target_profile.class_distribution or {}
        n_classes = len(class_dist)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Problem Type", "Classification")
        c2.metric("Classes", n_classes)
        c3.metric(
            "Imbalance Ratio",
            f"{target_profile.imbalance_ratio:.2f}"
            if target_profile.imbalance_ratio is not None
            else "-"
        )
        c4.metric("Samples", sum(class_dist.values()))

        if class_dist:
            st.markdown("##### Class Distribution")

            chart_df = (
                pd.DataFrame(
                    {
                        "Class": list(class_dist.keys()),
                        "Count": list(class_dist.values()),
                    }).sort_values("Count", ascending=False).set_index("Class"))

            st.bar_chart(chart_df)

        c1, c2 = st.columns(2)

        with c1:
            st.info(f"**Majority Class**\n\n" f"{target_profile.majority_class}")

        with c2:
            st.info(f"**Minority Class**\n\n" f"{target_profile.minority_class}")


