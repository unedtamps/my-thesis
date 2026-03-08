import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import (
    MODELS, MODEL_LABELS, SUBJECTS, SUBJECT_LABELS,
    load_csv, load_all_models, setup_sidebar, model_color_map,
)

st.set_page_config(page_title="Scenario 4: Keyword Trends", page_icon="📈", layout="wide")
st.title("📈 Scenario 4: Keyword Trend Evaluation")
st.markdown("""
Validates whether models capture statistically popular keywords and retain them over time.

**Ground Truth** — TF-IDF per year identifies important keywords classified by trend:
- 📈 *Emerging*: positive slope — growing importance over time
- 🔒 *Stable*: near-zero slope — consistently important
- 📉 *Decaying*: negative slope — declining importance

**SPAN** — longest consecutive years a keyword appears in a model's topic words.
A good model should have high SPAN for stable keywords and capture emerging trends early.

"""
)
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()

st.divider()

# ============================================================
# Ground Truth Keywords
# ============================================================
st.header("Ground Truth Keywords (TF-IDF)")

# Ground truth is the same for models sharing the same data source
# Just pick the first available
gt = None
for m in selected_models:
    gt = load_csv(m, "tren", subject, "ground_truth_keywords.csv")
    if gt is not None and len(gt) > 0:
        break

if gt is not None and len(gt) > 0:
    tab_em, tab_st, tab_dc = st.tabs(["Emerging", "Stable", "Decaying"])

    for tab, cat in [(tab_em, "emerging"), (tab_st, "stable"), (tab_dc, "decaying")]:
        with tab:
            cat_df = gt[gt["category"] == cat]
            if len(cat_df) > 0:
                st.dataframe(
                    cat_df[["word", "slope", "r_squared", "early_avg", "late_avg", "first_year", "n_present"]]
                    .reset_index(drop=True),
                    use_container_width=True,
                )
else:
    st.info("No ground truth keywords available.")

st.divider()

# ============================================================
# SPAN Comparison
# ============================================================
st.header("SPAN Analysis")

tab_summary, tab_detail, tab_heatmap = st.tabs(["Summary", "Per-Keyword", "Presence Heatmap"])

with tab_summary:
    span_all = load_all_models("tren", subject, "keyword_span_summary.csv")
    if len(span_all) > 0:
        span_f = span_all[span_all["model"].isin([MODEL_LABELS[m] for m in selected_models])]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                span_f, x="category", y="avg_span", color="model",
                barmode="group", color_discrete_map=colors,
                title="Average SPAN per Category",
                labels={"avg_span": "Avg SPAN (years)", "category": ""},
            )
            fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                span_f, x="category", y="avg_coverage_pct", color="model",
                barmode="group", color_discrete_map=colors,
                title="Average Coverage %",
                labels={"avg_coverage_pct": "Coverage %", "category": ""},
            )
            fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        # Never captured
        fig = px.bar(
            span_f, x="category", y="n_never_captured", color="model",
            barmode="group", color_discrete_map=colors,
            title="Keywords Never Captured (SPAN = 0)",
            labels={"n_never_captured": "Count", "category": ""},
        )
        fig.update_layout(height=350, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No SPAN summary data available.")

with tab_detail:
    model_select = st.selectbox(
        "Select model for detail view",
        selected_models, format_func=lambda x: MODEL_LABELS[x],
        key="span_detail_model",
    )
    span_df = load_csv(model_select, "tren", subject, "keyword_span.csv")
    if span_df is not None and len(span_df) > 0:
        cat_filter = st.selectbox("Category", ["all", "emerging", "stable", "decaying"], key="span_cat")
        if cat_filter != "all":
            span_df = span_df[span_df["category"] == cat_filter]

        display_cols = ["word", "category", "span", "total_years_present", "coverage_pct",
                        "total_topic_hits", "avg_topics_when_present", "presence"]
        available = [c for c in display_cols if c in span_df.columns]
        st.dataframe(
            span_df[available].sort_values("span", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

with tab_heatmap:
    model_hm = st.selectbox(
        "Select model for heatmap",
        selected_models, format_func=lambda x: MODEL_LABELS[x],
        key="heatmap_model",
    )
    span_hm = load_csv(model_hm, "tren", subject, "keyword_span.csv")
    if span_hm is not None and len(span_hm) > 0:
        cat_hm = st.selectbox("Category", ["emerging", "stable", "decaying"], key="hm_cat")
        cat_data = span_hm[span_hm["category"] == cat_hm].copy()

        if len(cat_data) > 0 and "topic_counts_per_year" in cat_data.columns:
            # Parse topic counts array
            import ast
            rows_parsed = []
            for _, r in cat_data.iterrows():
                try:
                    counts = ast.literal_eval(r["topic_counts_per_year"])
                    rows_parsed.append({"word": r["word"], "counts": counts})
                except:
                    pass

            if rows_parsed:
                # Determine years from the length
                n_years = len(rows_parsed[0]["counts"])
                years = list(range(2000, 2000 + n_years))

                matrix = []
                words_list = []
                for rp in rows_parsed:
                    matrix.append(rp["counts"])
                    words_list.append(rp["word"])

                matrix_np = np.array(matrix)

                fig = px.imshow(
                    matrix_np, x=[str(y) for y in years], y=words_list,
                    color_continuous_scale="YlOrRd",
                    title=f"{MODEL_LABELS[model_hm]}: {cat_hm.title()} Keywords — Topics per Year",
                    labels={"color": "# Topics"},
                    aspect="auto",
                )
                fig.update_layout(height=max(400, len(words_list) * 25))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No heatmap data available.")
