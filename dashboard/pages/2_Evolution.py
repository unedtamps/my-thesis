import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import (
    MODELS, MODEL_LABELS, SUBJECTS, SUBJECT_LABELS,
    load_csv, load_all_models, setup_sidebar, model_color_map,
)

st.set_page_config(page_title="Skenario 2: Evolution", page_icon="📊", layout="wide")
st.title("📊 Skenario 2: Temporal Evolution")
st.markdown("""
Analyzes how topics change over the 25-year timeline (2000–2025).

**Part A — Temporal Analysis**: Tracks topic prevalence, c-TF-IDF word evolution, 
and identifies growing/stable/declining topics via linear regression slope.

**Part B — Evolution Quality**: Measures how well topics maintain quality during transitions:
- **TTC** — cross-time coherence (topic words at *t* evaluated on corpus at *t+1*)
- **TTS** — term stability (cosine similarity of vocabulary *t* → *t+1*)
- **TTQ** — combined quality (TTC × TTS)

*See [Glossary](/Glossary) for detailed definitions.*
""")

subject, selected_models = setup_sidebar()
colors = model_color_map()

# ============================================================
# 2A: Temporal Analysis
# ============================================================
st.header("2A: Temporal Analysis")
st.divider()

# --- Per-year coherence + IRBO ---
tab1, tab2, tab3 = st.tabs(["Coherence & Diversity", "Topic Trends", "Topic Explorer"])

with tab1:
    pym = load_all_models("temporal", subject, "per_year_metrics.csv")
    if len(pym) > 0:
        pym_f = pym[pym["model"].isin([MODEL_LABELS[m] for m in selected_models])]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                pym_f, x="year", y="coherence_cv", color="model",
                color_discrete_map=colors,
                title="Coherence (C_v) per Year",
                labels={"coherence_cv": "C_v", "year": "Year"},
            )
            fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                pym_f, x="year", y="irbo_mean", color="model",
                color_discrete_map=colors,
                title="Diversity (IRBO) per Year",
                labels={"irbo_mean": "IRBO", "year": "Year"},
            )
            fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        # Topic quality
        fig = px.line(
            pym_f, x="year", y="topic_quality", color="model",
            color_discrete_map=colors,
            title="Topic Quality per Year",
            labels={"topic_quality": "Quality", "year": "Year"},
        )
        fig.update_layout(height=350, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No per-year metrics available.")

with tab2:
    # --- Trend distribution ---
    st.subheader("Trend Distribution")

    trend_data = []
    for m in selected_models:
        df = load_csv(m, "temporal", subject, "topic_trends.csv")
        if df is not None and len(df) > 0:
            for t in ["GROWING", "STABLE", "DECLINING"]:
                trend_data.append({
                    "model": MODEL_LABELS[m],
                    "trend": t,
                    "count": len(df[df["trend"] == t]),
                })

    if trend_data:
        td = pd.DataFrame(trend_data)
        fig = px.bar(
            td, x="model", y="count", color="trend", barmode="stack",
            color_discrete_map={"GROWING": "#22c55e", "STABLE": "#3b82f6", "DECLINING": "#ef4444"},
            title="Topic Trend Distribution",
            labels={"count": "Number of Topics", "model": ""},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # Top growing + declining per model
    for m in selected_models:
        df = load_csv(m, "temporal", subject, "topic_trends.csv")
        if df is None or len(df) == 0:
            continue

        with st.expander(f"📈 {MODEL_LABELS[m]} — Top Growing & Declining"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 5 Growing**")
                growing = df[df["trend"] == "GROWING"].nlargest(5, "slope")
                if len(growing) > 0:
                    st.dataframe(
                        growing[["topic_id", "top_words", "slope", "r_squared", "early_proportion", "late_proportion"]]
                        .reset_index(drop=True), use_container_width=True
                    )
            with col2:
                st.markdown("**Top 5 Declining**")
                declining = df[df["trend"] == "DECLINING"].nsmallest(5, "slope")
                if len(declining) > 0:
                    st.dataframe(
                        declining[["topic_id", "top_words", "slope", "r_squared", "early_proportion", "late_proportion"]]
                        .reset_index(drop=True), use_container_width=True
                    )

with tab3:
    st.subheader("Interactive Topic Explorer")
    st.caption("Select a model and topic to view its prevalence trend and word evolution over time.")

    explorer_model = st.selectbox(
        "Model", selected_models,
        format_func=lambda x: MODEL_LABELS[x],
        key="explorer_model",
    )

    # Load topic data
    prev_df = load_csv(explorer_model, "temporal", subject, "topic_prevalence.csv")
    word_df = load_csv(explorer_model, "temporal", subject, "topic_word_evolution.csv")
    trends_df = load_csv(explorer_model, "temporal", subject, "topic_trends.csv")

    if prev_df is not None and len(prev_df) > 0:
        topic_ids = sorted(prev_df["topic_id"].unique())

        # Build topic labels with trend + top words
        topic_labels = {}
        for tid in topic_ids:
            label = f"Topic {tid}"
            if trends_df is not None and len(trends_df) > 0:
                t_row = trends_df[trends_df["topic_id"] == tid]
                if len(t_row) > 0:
                    trend = t_row.iloc[0].get("trend", "")
                    words = t_row.iloc[0].get("top_words", "")
                    icon = {"GROWING": "📈", "DECLINING": "📉", "STABLE": "🔒"}.get(trend, "")
                    label = f"{icon} Topic {tid}: {words[:60]}"
            topic_labels[tid] = label

        selected_topic = st.selectbox(
            "Topic",
            topic_ids,
            format_func=lambda x: topic_labels.get(x, f"Topic {x}"),
            key="explorer_topic",
        )

        # --- Prevalence Chart ---
        topic_prev = prev_df[prev_df["topic_id"] == selected_topic].sort_values("year")
        if len(topic_prev) > 0:
            # Get trend info
            trend_info = ""
            if trends_df is not None and len(trends_df) > 0:
                t_row = trends_df[trends_df["topic_id"] == selected_topic]
                if len(t_row) > 0:
                    tr = t_row.iloc[0]
                    trend_info = f"**{tr.get('trend', '')}** — slope: {tr.get('slope', 0):.6f}, R²: {tr.get('r_squared', 0):.4f}"

            col_chart, col_info = st.columns([3, 1])
            with col_chart:
                fig = px.area(
                    topic_prev, x="year", y="proportion",
                    title=f"Topic {selected_topic} — Prevalence Over Time",
                    labels={"proportion": "Proportion of docs", "year": "Year"},
                )
                color = colors.get(MODEL_LABELS[explorer_model], "#3b82f6")
                # Convert hex to rgba for fill transparency
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fig.update_traces(
                    line_color=color,
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col_info:
                st.markdown("#### Stats")
                if trend_info:
                    st.markdown(trend_info)
                st.metric("Total Docs", int(topic_prev["doc_count"].sum()))
                st.metric("Peak Year", int(topic_prev.loc[topic_prev["proportion"].idxmax(), "year"]))
                st.metric("Years Active", len(topic_prev[topic_prev["doc_count"] > 0]))

        # --- Word Evolution Table ---
        st.markdown("#### Word Evolution Over Time")
        if word_df is not None and len(word_df) > 0:
            topic_words = word_df[word_df["topic_id"] == selected_topic].sort_values("year")
            if len(topic_words) > 0:
                display_words = topic_words[["year", "top_words"]].copy()
                display_words.columns = ["Year", "Top Words"]
                st.dataframe(display_words.reset_index(drop=True), use_container_width=True, height=400)
            else:
                st.info("No word evolution data for this topic.")
        else:
            st.info("No word evolution data available.")
    else:
        st.info("No topic prevalence data available.")
# 2B: Evolution Quality
# ============================================================
st.header("2B: Evolution Quality (TTC · TTS · TTQ)")
st.divider()

trans = load_all_models("evolution", subject, "transition_summary.csv")
if len(trans) > 0:
    trans_f = trans[trans["model"].isin([MODEL_LABELS[m] for m in selected_models])]

    tab_ttc, tab_tts, tab_ttq = st.tabs(["TTC", "TTS", "TTQ"])

    with tab_ttc:
        fig = px.line(
            trans_f, x="year_from", y="avg_ttc", color="model",
            color_discrete_map=colors,
            title="TTC (Cross-Time Coherence) Over Transitions",
            labels={"avg_ttc": "Avg TTC", "year_from": "Year"},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with tab_tts:
        fig = px.line(
            trans_f, x="year_from", y="avg_tts", color="model",
            color_discrete_map=colors,
            title="TTS (Term Stability) Over Transitions",
            labels={"avg_tts": "Avg TTS", "year_from": "Year"},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with tab_ttq:
        fig = px.line(
            trans_f, x="year_from", y="avg_ttq", color="model",
            color_discrete_map=colors,
            title="TTQ (Topic Transition Quality) Over Transitions",
            labels={"avg_ttq": "Avg TTQ", "year_from": "Year"},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # Summary comparison
    st.subheader("Overall Evolution Summary")
    evo_all = load_all_models("evolution", subject, "evolution_summary.csv")
    if len(evo_all) > 0:
        evo_f = evo_all[evo_all["model"].isin([MODEL_LABELS[m] for m in selected_models])]
        display = ["model", "avg_ttc", "avg_tts", "avg_ttq"]
        available = [c for c in display if c in evo_f.columns]
        st.dataframe(evo_f[available].set_index("model"), use_container_width=True)
else:
    st.info("No evolution data available.")
