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

st.set_page_config(page_title="Scenario 2: Timestep", page_icon="📊", layout="wide")
st.title("📊 Scenario 2: Timestep")
st.markdown("""
Kualitas Topik per Titik Waktu — menganalisis seberapa baik kualitas topik di setiap tahun selama rentang 2000–2025.

- **Coherence (C_v)**: koherensi semantik topik per tahun
- **Diversity (IRBO)**: keragaman kata antar topik per tahun
- **Topic Quality**: harmonic mean dari C_v dan IRBO
- **Topic Trends**: klasifikasi topik menjadi GROWING / STABLE / DECLINING berdasarkan regresi linier
""")
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()

CHART_FONT = dict(family="Arial Black, Arial, sans-serif", size=15)
CHART_TITLE_FONT = dict(family="Arial Black, Arial, sans-serif", size=15)
AXIS_FONT = dict(family="Arial Black, Arial, sans-serif", size=15)

# ============================================================
# Per-year metrics
# ============================================================
st.header("Kualitas Topik per Tahun")
st.divider()

tab1, tab_metrics, tab2 = st.tabs(["Coherence & Diversity", "Metrics Table", "Topic Trends"])

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
            fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15),
                              font=CHART_FONT, title_font=CHART_TITLE_FONT,
                              xaxis=dict(tickmode="array", tickvals=[2000, 2005, 2010, 2015, 2020, 2025], range=[2000, 2026]))
            fig.update_xaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
            fig.update_yaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                pym_f, x="year", y="irbo_mean", color="model",
                color_discrete_map=colors,
                title="Diversity (IRBO) per Year",
                labels={"irbo_mean": "IRBO", "year": "Year"},
            )
            fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15),
                              font=CHART_FONT, title_font=CHART_TITLE_FONT,
                              xaxis=dict(tickmode="array", tickvals=[2000, 2005, 2010, 2015, 2020, 2025], range=[2000, 2026]))
            fig.update_xaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
            fig.update_yaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            pym_f, x="year", y="topic_quality", color="model",
            color_discrete_map=colors,
            title="Topic Quality per Year",
            labels={"topic_quality": "Quality", "year": "Year"},
        )
        fig.update_layout(height=350, legend=dict(orientation="h", y=-0.15),
                          font=CHART_FONT, title_font=CHART_TITLE_FONT,
                          xaxis=dict(tickmode="array", tickvals=[2000, 2005, 2010, 2015, 2020, 2025], range=[2000, 2026]))
        fig.update_xaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        fig.update_yaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No per-year metrics available.")

with tab_metrics:
    pym = load_all_models("temporal", subject, "per_year_metrics.csv")
    if len(pym) > 0:
        pym_f = pym[pym["model"].isin([MODEL_LABELS[m] for m in selected_models])]
        st.subheader("Yearly Metrics per Model")
        for m in selected_models:
            m_label = MODEL_LABELS[m]
            m_df = pym_f[pym_f["model"] == m_label]
            if not m_df.empty:
                with st.expander(f"📊 {m_label}"):
                    display_df = m_df[["year", "coherence_cv", "irbo_mean", "topic_quality"]].copy()
                    display_df = display_df.rename(columns={
                        "coherence_cv": "C_v",
                        "irbo_mean": "Diversity (IRBO)",
                        "topic_quality": "Quality"
                    }).set_index("year")
                    st.dataframe(display_df, use_container_width=True)

                    st.divider()
                    st.markdown("**Average Over Time**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg C_v", f"{display_df['C_v'].mean():.4f}")
                    col2.metric("Avg Diversity", f"{display_df['Diversity (IRBO)'].mean():.4f}")
                    col3.metric("Avg Quality", f"{display_df['Quality'].mean():.4f}")
    else:
        st.info("No per-year metrics available.")

with tab2:
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
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15),
                          font=CHART_FONT, title_font=CHART_TITLE_FONT)
        fig.update_xaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        fig.update_yaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        st.plotly_chart(fig, use_container_width=True)

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

st.info("💡 For per-topic deep-dives (prevalence, drift, continuity), visit the **[Topic Explorer](/5_Topic_Explorer)** page.")
