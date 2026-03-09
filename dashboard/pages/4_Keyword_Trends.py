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

st.divider()

# ============================================================
# Paper-Faithful avg-SPAN (Gupta et al., 2018)
# ============================================================
st.header("Paper-Faithful avg-SPAN (Gupta et al., 2018)")
st.markdown("""
Following the SPAN metric from *Deep Temporal-Recurrent-Replicated-Softmax for Topical Trends over Time*
([arXiv:1711.05626v2](https://arxiv.org/abs/1711.05626)):

- **keyword-trend** — binary presence/absence of a word in **any** discovered topic per year
- **SPAN (Sₖ)** — longest consecutive years the keyword appears
- **v̂ₖ** — total count of keyword across all topics and years
- **Sₖᵈⁱᶜᵗ = Sₖ / v̂ₖ** — frequency-normalized SPAN
- **avg-SPAN = (1/||Q̂||) × Σ Sₖᵈⁱᶜᵗ** — averaged over **all** unique topic-terms

Higher avg-SPAN → model better captures trending keywords over time.
""")

tab_paper_summary, tab_paper_detail, tab_paper_top = st.tabs(
    ["Model Comparison", "Per-Term Detail", "Top SPAN Terms"]
)

with tab_paper_summary:
    paper_all = load_all_models("tren", subject, "keyword_span_paper_summary.csv")
    if len(paper_all) > 0:
        paper_f = paper_all[paper_all["model"].isin([MODEL_LABELS[m] for m in selected_models])]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                paper_f, x="model", y="avg_span_paper", color="model",
                color_discrete_map=colors,
                title="avg-SPAN (Paper Formula, Frequency-Weighted)",
                labels={"avg_span_paper": "avg-SPAN", "model": ""},
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                paper_f, x="model", y="avg_span_simple", color="model",
                color_discrete_map=colors,
                title="avg-SPAN (Simple Mean)",
                labels={"avg_span_simple": "avg-SPAN (simple)", "model": ""},
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Summary metrics table
        display_cols_paper = ["model", "total_unique_terms", "avg_span_paper",
                              "avg_span_simple", "total_corpus_freq",
                              "terms_not_in_corpus"]
        available_paper = [c for c in display_cols_paper if c in paper_f.columns]
        st.dataframe(
            paper_f[available_paper].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "model": "Model",
                "total_unique_terms": "||Q̂|| (unique terms)",
                "avg_span_paper": st.column_config.NumberColumn("avg-SPAN (paper)", format="%.6f"),
                "avg_span_simple": st.column_config.NumberColumn("avg-SPAN (simple)", format="%.4f"),
                "total_corpus_freq": "Σv̂ₖ (corpus freq)",
                "terms_not_in_corpus": "Terms not in corpus",
            },
        )
    else:
        st.info("No paper-faithful SPAN data. Run the updated keyword_trend notebooks first.")

with tab_paper_detail:
    model_paper = st.selectbox(
        "Select model", selected_models,
        format_func=lambda x: MODEL_LABELS[x],
        key="paper_detail_model",
    )
    paper_df = load_csv(model_paper, "tren", subject, "keyword_span_paper.csv")
    if paper_df is not None and len(paper_df) > 0:
        sort_by = st.selectbox("Sort by", ["span", "v_hat", "s_dict", "word"], key="paper_sort")
        asc = sort_by == "word"
        display_cols_d = ["word", "span", "v_hat", "s_dict", "years_present", "coverage_pct"]
        available_d = [c for c in display_cols_d if c in paper_df.columns]
        st.dataframe(
            paper_df[available_d].sort_values(sort_by, ascending=asc).head(100).reset_index(drop=True),
            use_container_width=True,
            column_config={
                "word": "Keyword",
                "span": "SPAN (Sₖ)",
                "v_hat": "v̂ₖ (frequency)",
                "s_dict": st.column_config.NumberColumn("Sₖᵈⁱᶜᵗ", format="%.4f"),
                "years_present": "Years Present",
                "coverage_pct": st.column_config.NumberColumn("Coverage %", format="%.1f%%"),
            },
        )
        st.caption(f"Showing top 100 of {len(paper_df)} terms")
    else:
        st.info("No per-term data available.")

with tab_paper_top:
    model_top = st.selectbox(
        "Select model", selected_models,
        format_func=lambda x: MODEL_LABELS[x],
        key="paper_top_model",
    )
    paper_top_df = load_csv(model_top, "tren", subject, "keyword_span_paper.csv")
    if paper_top_df is not None and len(paper_top_df) > 0 and "keyword_trend" in paper_top_df.columns:
        import ast
        n_top = st.slider("Number of top keywords", 10, 50, 20, key="paper_top_n")
        top_terms = paper_top_df.nlargest(n_top, "span")

        rows_hm = []
        for _, r in top_terms.iterrows():
            try:
                trend = ast.literal_eval(r["keyword_trend"])
                rows_hm.append({"word": r["word"], "trend": trend, "span": r["span"]})
            except:
                pass

        if rows_hm:
            n_years = len(rows_hm[0]["trend"])
            years = list(range(2000, 2000 + n_years))
            matrix = [row["trend"] for row in rows_hm]
            words = [f"{row['word']} (S={row['span']})" for row in rows_hm]

            fig = px.imshow(
                np.array(matrix), x=[str(y) for y in years], y=words,
                color_continuous_scale=[[0, "#1e1e2e"], [1, "#22c55e"]],
                title=f"{MODEL_LABELS[model_top]}: Keyword-Trend (Top {n_top} by SPAN)",
                labels={"color": "Present"},
                aspect="auto",
            )
            fig.update_layout(height=max(400, len(words) * 28))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No keyword-trend data available.")

st.divider()

# ============================================================
# Shared Keyword SPAN — Cross-Model Comparison (Paper Table 6)
# ============================================================
st.header("Shared Keyword SPAN — Cross-Model Comparison")
st.markdown("""
Like Table 6 and Figure 4 in the paper: the **same** top-50 corpus keywords are evaluated
against **all** models, enabling a direct apple-to-apple SPAN comparison.
""")

# Load shared data helper
def load_shared_csv(subject, filename):
    from utils.data_loader import RESULTS_DIR
    path = RESULTS_DIR / "shared" / "tren" / subject / filename
    if path.exists() and path.stat().st_size > 10:
        return pd.read_csv(path)
    return None

tab_shared_summary, tab_shared_table, tab_shared_trend = st.tabs(
    ["avg-SPAN Comparison", "Per-Keyword Table (Table 6)", "Keyword-Trend (Figure 4)"]
)

with tab_shared_summary:
    shared_summary = load_shared_csv(subject, "shared_keyword_span_summary.csv")
    if shared_summary is not None and len(shared_summary) > 0:
        shared_f = shared_summary[shared_summary["model"].isin(
            [MODEL_LABELS[m] for m in selected_models]
        )]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                shared_f, x="model", y="avg_span_paper", color="model",
                color_discrete_map=colors,
                title="avg-SPAN — Paper Formula: (1/N) × Σ (Sₖ/v̂ₖ)",
                labels={"avg_span_paper": "avg-SPAN (paper)", "model": ""},
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                shared_f, x="model", y="avg_span_simple", color="model",
                color_discrete_map=colors,
                title="avg-SPAN — Simple Mean (years)",
                labels={"avg_span_simple": "avg SPAN (years)", "model": ""},
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            shared_f[["model", "n_keywords", "avg_span_paper", "avg_span_simple",
                       "n_captured", "n_full_span"]].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "model": "Model",
                "n_keywords": "Shared Keywords",
                "avg_span_paper": st.column_config.NumberColumn("avg-SPAN (paper)", format="%.6f"),
                "avg_span_simple": st.column_config.NumberColumn("avg-SPAN (simple)", format="%.2f"),
                "n_captured": "Captured",
                "n_full_span": "Full SPAN",
            },
        )
    else:
        st.info("No shared keyword data. Run `notebooks/models/shared_keyword_span.py` first.")

with tab_shared_table:
    shared_df = load_shared_csv(subject, "shared_keyword_span.csv")
    if shared_df is not None and len(shared_df) > 0:
        # Build side-by-side SPAN table (like paper Table 6)
        span_cols = ["word", "v_hat"]
        for m in selected_models:
            lbl = MODEL_LABELS[m]
            if f"span_{lbl}" in shared_df.columns:
                span_cols.append(f"span_{lbl}")

        sort_col = st.selectbox("Sort by", ["v_hat"] + [f"span_{MODEL_LABELS[m]}" for m in selected_models],
                                key="shared_sort", format_func=lambda x: x.replace("span_", "SPAN ").replace("v_hat", "Corpus freq (v̂ₖ)"))

        st.dataframe(
            shared_df[span_cols].sort_values(sort_col, ascending=False).reset_index(drop=True),
            use_container_width=True,
            column_config={
                "word": "Keyword",
                "v_hat": "v̂ₖ (corpus freq)",
                **{f"span_{MODEL_LABELS[m]}": f"SPAN ({MODEL_LABELS[m]})" for m in selected_models},
            },
        )
    else:
        st.info("No shared keyword data available.")

with tab_shared_trend:
    shared_df2 = load_shared_csv(subject, "shared_keyword_span.csv")
    if shared_df2 is not None and len(shared_df2) > 0:
        import ast as _ast
        model_trend = st.selectbox(
            "Select model for keyword-trend",
            selected_models, format_func=lambda x: MODEL_LABELS[x],
            key="shared_trend_model",
        )
        n_kw = st.slider("Number of keywords", 10, 50, 20, key="shared_n_kw")
        lbl = MODEL_LABELS[model_trend]
        trend_col = f"trend_{lbl}"
        span_col = f"span_{lbl}"

        if trend_col in shared_df2.columns:
            top_kw = shared_df2.nlargest(n_kw, span_col)
            rows_hm = []
            for _, r in top_kw.iterrows():
                try:
                    trend = _ast.literal_eval(r[trend_col])
                    rows_hm.append({"word": r["word"], "trend": trend, "span": r[span_col]})
                except:
                    pass

            if rows_hm:
                n_years = len(rows_hm[0]["trend"])
                years = list(range(2000, 2000 + n_years))
                matrix = [row["trend"] for row in rows_hm]
                words = [f"{row['word']} (S={row['span']})" for row in rows_hm]

                fig = px.imshow(
                    np.array(matrix), x=[str(y) for y in years], y=words,
                    color_continuous_scale=[[0, "#1e1e2e"], [1, "#22c55e"]],
                    title=f"Keyword-Trend: {lbl} — Top {n_kw} Shared Keywords by SPAN",
                    labels={"color": "Present"},
                    aspect="auto",
                )
                fig.update_layout(height=max(400, len(words) * 28))
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No keyword-trend data available.")
