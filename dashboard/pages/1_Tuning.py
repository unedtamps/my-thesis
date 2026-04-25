import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import (
    MODELS, MODEL_LABELS, SUBJECTS, SUBJECT_LABELS,
    load_tuning, setup_sidebar, model_color_map,
)

st.set_page_config(page_title="Scenario 1: Tuning", page_icon="🔧", layout="wide")
st.title("🔧 Scenario 1: Hyperparameter Tuning")
st.markdown("""
Grid search over hyperparameters to find the best model configuration per subject.
Each configuration is scored by three metrics:
- **Coherence (C_v)** — semantic relatedness of words within topics
- **IRBO** — diversity between topics (low overlap)
- **Topic Quality** — harmonic mean of coherence and IRBO

The best model balances both coherence and diversity.
"""
)
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()

st.divider()

# --- Best config per model ---
st.subheader("Best Configuration per Model")
best_rows = []
for m in selected_models:
    df = load_tuning(m, subject)
    if df is None or len(df) == 0 or "topic_quality" not in df.columns:
        continue
    best = df.loc[df["topic_quality"].idxmax()].to_dict()
    best["model"] = MODEL_LABELS[m]
    best_rows.append(best)

if best_rows:
    best_df = pd.DataFrame(best_rows)
    display_cols = ["model", "topic_quality", "coherence_cv", "irbo_mean"]
    # Add model-specific param columns if they exist
    for col in best_df.columns:
        if col not in display_cols and col not in ["subject", "time_seconds", "workers", "random_state", "seed"]:
            display_cols.append(col)
    available = [c for c in display_cols if c in best_df.columns]
    st.dataframe(best_df[available].set_index("model"), use_container_width=True)
else:
    st.info("No tuning data available for selected models.")

st.divider()

# --- Per-model exploration ---
for m in selected_models:
    label = MODEL_LABELS[m]
    df = load_tuning(m, subject)
    if df is None or len(df) == 0 or "topic_quality" not in df.columns:
        continue

    with st.expander(f"📊 {label} — Detailed Results ({len(df)} configs)", expanded=len(selected_models) == 1):
        col1, col2 = st.columns(2)

        with col1:
            # Coherence vs IRBO scatter
            if "irbo_mean" in df.columns:
                fig = px.scatter(
                    df, x="coherence_cv", y="irbo_mean",
                    size="topic_quality", color="topic_quality",
                    color_continuous_scale="Viridis",
                    hover_data=["num_topics"] if "num_topics" in df.columns else None,
                    title=f"{label}: Coherence vs Diversity",
                    labels={"coherence_cv": "Coherence (C_v)", "irbo_mean": "IRBO Diversity"},
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Topic quality by number of topics
            if "num_topics" in df.columns:
                fig = px.box(
                    df, x="num_topics", y="topic_quality",
                    title=f"{label}: Quality by Number of Topics",
                    labels={"num_topics": "Number of Topics", "topic_quality": "Topic Quality"},
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Top 30 configs table
        st.markdown("**Top 30 Configs (by Topic Quality)**")
        top30 = df.nlargest(30, "topic_quality")
        hide = ["subject", "random_state", "workers", "seed"]
        display_cols = [c for c in top30.columns if c not in hide]
        st.dataframe(top30[display_cols].reset_index(drop=True), use_container_width=True)
