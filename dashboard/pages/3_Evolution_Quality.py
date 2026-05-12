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

st.set_page_config(page_title="Scenario 3: Evolution Quality", page_icon="🔄", layout="wide")
st.title("🔄 Scenario 3: Evolution")
st.markdown("""
Evaluates how well topics maintain their quality when transitioning from one year to the next.

- **TTC** — Temporal Topic Coherence: normalized NPMI of cross-time word pairs (*words_t* × *words_t+1*) evaluated against the full corpus
- **TTS** — Temporal Topic Stability: RBO (Rank-Biased Overlap) similarity of topic words between *t* and *t+1*
- **TTQ** — Temporal Topic Quality: harmonic mean of TTC and TTS
""")
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()

CHART_FONT = dict(family="Arial Black, Arial, sans-serif", size=15)
CHART_TITLE_FONT = dict(family="Arial Black, Arial, sans-serif", size=15)
AXIS_FONT = dict(family="Arial Black, Arial, sans-serif", size=15)

# ============================================================
# Evolution Quality: TTC · TTS · TTQ
# ============================================================
st.header("Topic Evolution Quality (TTC · TTS · TTQ)")
st.divider()

trans = load_all_models("evolution", subject, "transition_summary.csv")
if len(trans) > 0:
    trans_f = trans[trans["model"].isin([MODEL_LABELS[m] for m in selected_models])]

    tab_ttc, tab_tts, tab_ttq, tab_metrics_2b = st.tabs(["TTC", "TTS", "TTQ", "Metrics Table"])

    with tab_ttc:
        fig = px.line(
            trans_f, x="year_from", y="avg_ttc", color="model",
            color_discrete_map=colors,
            title="TTC (Cross-Time Coherence) Over Transitions",
            labels={"avg_ttc": "Avg TTC", "year_from": "Year"},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15),
                          font=CHART_FONT, title_font=CHART_TITLE_FONT,
                          xaxis=dict(tickmode="array", tickvals=[2000, 2005, 2010, 2015, 2020, 2025], range=[2000, 2025]))
        fig.update_xaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        fig.update_yaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        st.plotly_chart(fig, use_container_width=True)

    with tab_tts:
        fig = px.line(
            trans_f, x="year_from", y="avg_tts", color="model",
            color_discrete_map=colors,
            title="TTS (Term Stability) Over Transitions",
            labels={"avg_tts": "Avg TTS", "year_from": "Year"},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15),
                          font=CHART_FONT, title_font=CHART_TITLE_FONT,
                          xaxis=dict(tickmode="array", tickvals=[2000, 2005, 2010, 2015, 2020, 2025], range=[2000, 2025]))
        fig.update_xaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        fig.update_yaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        st.plotly_chart(fig, use_container_width=True)

    with tab_ttq:
        fig = px.line(
            trans_f, x="year_from", y="avg_ttq", color="model",
            color_discrete_map=colors,
            title="TTQ (Topic Transition Quality) Over Transitions",
            labels={"avg_ttq": "Avg TTQ", "year_from": "Year"},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15),
                          font=CHART_FONT, title_font=CHART_TITLE_FONT,
                          xaxis=dict(tickmode="array", tickvals=[2000, 2005, 2010, 2015, 2020, 2025], range=[2000, 2025]))
        fig.update_xaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        fig.update_yaxes(tickfont=AXIS_FONT, title_font=AXIS_FONT)
        st.plotly_chart(fig, use_container_width=True)

    with tab_metrics_2b:
        st.subheader("TTC · TTS · TTQ per Year — All Models")
        for m in selected_models:
            m_label = MODEL_LABELS[m]
            m_df = trans_f[trans_f["model"] == m_label][["year_from", "avg_ttc", "avg_tts", "avg_ttq"]].copy()
            if m_df.empty:
                continue
            with st.expander(f"📊 {m_label}"):
                display_df = m_df.rename(columns={
                    "avg_ttc": "TTC",
                    "avg_tts": "TTS",
                    "avg_ttq": "TTQ",
                }).set_index("year_from")
                display_df.index.name = "Year"
                st.dataframe(
                    display_df.style.format("{:.4f}", na_rep="—"),
                    use_container_width=True,
                )

                st.divider()
                st.markdown("**Average Over Time**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg TTC", f"{display_df['TTC'].mean():.4f}")
                col2.metric("Avg TTS", f"{display_df['TTS'].mean():.4f}")
                col3.metric("Avg TTQ", f"{display_df['TTQ'].mean():.4f}")

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
