import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import (
    MODELS, MODEL_LABELS, SUBJECTS, SUBJECT_LABELS,
    load_csv, load_tuning, load_all_models, setup_sidebar, model_color_map, RESULTS_DIR,
)

st.set_page_config(
    page_title="Topic Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Header ---
st.title("📊 Topic Modeling Dashboard")
st.markdown(
    "Comparing **LDA**, **DTM**, **BERTopic**, and **TopicGPT** on arXiv papers (2000–2025)"
)

subject, selected_models = setup_sidebar()
colors = model_color_map()

st.divider()

# --- Overview cards ---
st.subheader(f"Overview — {SUBJECT_LABELS[subject]}")

cols = st.columns(len(selected_models))
for i, m in enumerate(selected_models):
    label = MODEL_LABELS[m]
    with cols[i]:
        st.markdown(f"### {label}")

        # Best tuning result (normalized columns)
        tuning = load_tuning(m, subject)
        if tuning is not None and len(tuning) > 0 and "topic_quality" in tuning.columns:
            best = tuning.loc[tuning["topic_quality"].idxmax()]
            st.metric("Best Topic Quality", f"{best['topic_quality']:.4f}")
            if "coherence_cv" in best.index:
                st.metric("Coherence (C_v)", f"{best['coherence_cv']:.4f}")
            if "irbo_mean" in best.index:
                st.metric("Diversity (IRBO)", f"{best['irbo_mean']:.4f}")
        else:
            st.info("No tuning data")

        # Evolution summary
        evo = load_csv(m, "evolution", subject, "evolution_summary.csv")
        if evo is not None and len(evo) > 0:
            row = evo.iloc[0]
            st.metric("Avg Temporal Topic Quality (TTQ)", f"{row.get('avg_ttq', 0):.4f}")

        # Trends summary
        trends = load_csv(m, "temporal", subject, "topic_trends.csv")
        if trends is not None and len(trends) > 0:
            g = len(trends[trends["trend"] == "GROWING"])
            s = len(trends[trends["trend"] == "STABLE"])
            d = len(trends[trends["trend"] == "DECLINING"])
            st.caption(f"Growing: {g} · Stable: {s} · Declining: {d}")

st.divider()

# --- Cross-model comparison: Per-year coherence ---
st.subheader("Average Topic Quality per Year")
pym = load_all_models("temporal", subject, "per_year_metrics.csv")
if len(pym) > 0:
    pym_filtered = pym[pym["model"].isin([MODEL_LABELS[m] for m in selected_models])]
    
    if "topic_quality" in pym_filtered.columns:
        y_col = "topic_quality"
        y_label = "Avg Topic Quality"
    else:
        y_col = "coherence_cv"
        y_label = "Coherence (C_v)"

    fig = px.line(
        pym_filtered, x="year", y=y_col, color="model",
        color_discrete_map=colors,
        labels={y_col: y_label, "year": "Year", "model": "Model"},
    )
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No per-year metrics data available.")

# --- Cross-model comparison: Evolution quality ---
st.subheader("Evolution Quality (Avg Temporal Topic Quality) Over Time")
trans = load_all_models("evolution", subject, "transition_summary.csv")
if len(trans) > 0:
    trans_filtered = trans[trans["model"].isin([MODEL_LABELS[m] for m in selected_models])]
    fig = px.line(
        trans_filtered, x="year_from", y="avg_ttq", color="model",
        color_discrete_map=colors,
        labels={"avg_ttq": "Avg Temporal Topic Quality (TTQ)", "year_from": "Year", "model": "Model"},
    )
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No evolution data available.")