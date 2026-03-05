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
    st.caption("Each model is collapsible with its own search and topic selector.")

    for m in selected_models:
        label = MODEL_LABELS[m]
        m_prev = load_csv(m, "temporal", subject, "topic_prevalence.csv")
        m_trends = load_csv(m, "temporal", subject, "topic_trends.csv")
        m_words = load_csv(m, "temporal", subject, "topic_word_evolution.csv")

        if m_prev is None or len(m_prev) == 0:
            continue

        with st.expander(f"📊 {label}", expanded=(m == selected_models[0])):
            topic_ids = sorted(m_prev["topic_id"].unique())

            # Build labels
            t_labels = {}
            t_words = {}
            for tid in topic_ids:
                lbl = f"Topic {tid}"
                words = ""
                if m_trends is not None and len(m_trends) > 0:
                    t_row = m_trends[m_trends["topic_id"] == tid]
                    if len(t_row) > 0:
                        trend = t_row.iloc[0].get("trend", "")
                        words = str(t_row.iloc[0].get("top_words", ""))
                        icon = {"GROWING": "📈", "DECLINING": "📉", "STABLE": "🔒"}.get(trend, "")
                        lbl = f"{icon} T{tid}: {words[:55]}"
                t_labels[tid] = lbl
                t_words[tid] = words

            # Per-model search
            search = st.text_input("🔍 Search", placeholder="e.g. neural, quantum...", key=f"search_{m}")
            if search:
                filtered = [t for t in topic_ids if search.lower() in t_words[t].lower()]
                if not filtered:
                    st.warning(f"No match — showing all")
                    filtered = topic_ids
            else:
                filtered = topic_ids

            # Per-model topic multi-select
            picks = st.multiselect(
                "Topics", filtered,
                default=filtered[:2] if len(filtered) >= 2 else filtered[:1],
                format_func=lambda x: t_labels.get(x, f"Topic {x}"),
                key=f"topics_{m}",
            )

            if not picks:
                st.info("Select at least one topic.")
                continue

            # Chart with selected topics
            fig = go.Figure()
            for i, tid in enumerate(picks):
                tp = m_prev[m_prev["topic_id"] == tid].sort_values("year")
                if len(tp) > 0:
                    fig.add_trace(go.Scatter(
                        x=tp["year"], y=tp["proportion"],
                        name=f"T{tid}",
                        mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(size=4),
                    ))
            fig.update_layout(
                title=f"{label} — Topic Prevalence",
                xaxis_title="Year", yaxis_title="Proportion",
                height=380, legend=dict(orientation="h", y=-0.15),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Topic detail expanders
            for tid in picks:
                trend_tag = ""
                if m_trends is not None and len(m_trends) > 0:
                    t_row = m_trends[m_trends["topic_id"] == tid]
                    if len(t_row) > 0:
                        trend_tag = t_row.iloc[0].get("trend", "")
                icon = {"GROWING": "📈", "DECLINING": "📉", "STABLE": "🔒"}.get(trend_tag, "")

                with st.expander(f"{icon} Topic {tid} details"):
                    if m_trends is not None and len(m_trends) > 0:
                        t_row = m_trends[m_trends["topic_id"] == tid]
                        if len(t_row) > 0:
                            tr = t_row.iloc[0]
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Trend", f"{icon} {tr.get('trend', 'N/A')}")
                            c2.metric("Slope", f"{tr.get('slope', 0):.6f}")
                            c3.metric("R²", f"{tr.get('r_squared', 0):.4f}")

                    if m_words is not None and len(m_words) > 0:
                        tw = m_words[m_words["topic_id"] == tid].sort_values("year")
                        if len(tw) > 0:
                            display = tw[["year", "top_words"]].copy()
                            display.columns = ["Year", "Top Words"]
                            st.dataframe(display.reset_index(drop=True), use_container_width=True, height=250)
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
