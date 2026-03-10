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

1. **Paper-Faithful avg-SPAN**: Evaluates models on *all* unique topic-terms they discover, rewarding terms with high SPAN and punishing generic terms using their corpus frequency (v̂ₖ).
2. **Shared Keyword SPAN**: Compares all models on the *same* pool of top intersection keywords to directly evaluate which model retains common jargon better.
3. **Trend Category SPAN (TF-IDF)**: Compares performance specifically on 90 keywords separated by their TF-IDF linear regression slope over time (Emerging 📈, Stable 🔒, Decaying 📉).

A good model should capture more shared keywords, show high SPAN for stable keywords, and capture emerging trends early.
"""
)
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()

st.divider()

# ============================================================
# Paper-Faithful avg-SPAN (Gupta et al., 2018)
# ============================================================
st.header("1. Paper-Faithful avg-SPAN")
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
st.header("2. Shared Keyword SPAN (Cross-Model Comparison)")
st.markdown("""
Like Table 6 and Figure 4 in the paper: the **same** pool of intersection keywords (captured by at least 2 models) are evaluated
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
                       "n_captured", "capture_pct", "n_full_span"]].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "model": "Model",
                "n_keywords": "Total Candidates",
                "avg_span_paper": st.column_config.NumberColumn("avg-SPAN (paper)", format="%.6f"),
                "avg_span_simple": st.column_config.NumberColumn("avg-SPAN (simple)", format="%.2f"),
                "n_captured": "Captured",
                "capture_pct": st.column_config.NumberColumn("Captured (%)", format="%.1f%%"),
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

        st.markdown(f"**Total Intersection Keywords:** {len(shared_df):,} terms")
        
        col1, col2 = st.columns(2)
        with col1:
            sort_col = st.selectbox("Sort by", ["v_hat"] + [f"span_{MODEL_LABELS[m]}" for m in selected_models],
                                    key="shared_sort", format_func=lambda x: x.replace("span_", "SPAN ").replace("v_hat", "Corpus freq (v̂ₖ)"))
        with col2:
            limit_rows = st.selectbox("Show top N rows", [50, 100, 500, 1000, "All"], key="shared_limit")

        df_show = shared_df[span_cols].sort_values(sort_col, ascending=False).reset_index(drop=True)
        if limit_rows != "All":
            df_show = df_show.head(limit_rows)

        st.dataframe(
            df_show,
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
# ============================================================
# Trend Category SPAN (TF-IDF)
# ============================================================
st.header("3. Trend Category SPAN (TF-IDF)")
st.markdown("""
Evaluate model performance against **specific trend categories** determined by TF-IDF slope over the corpus.
For each subject, we identify 90 ground-truth keywords:
- 📈 **30 Emerging**: growing importance over time (highest positive slope)
- 🔒 **30 Stable**: consistently important (near-zero slope + high avg TF-IDF)
- 📉 **30 Decaying**: declining importance (highest negative slope)

Models are evaluated using the paper's frequency-weighted formula onto these specific 90 keywords.
""")

def load_shared_csv(target_subject, filename):
    from utils.data_loader import RESULTS_DIR
    path = RESULTS_DIR / "shared" / "tren" / target_subject / filename
    if path.exists() and path.stat().st_size > 10:
        return pd.read_csv(path)
    return None

tc_sum_df = load_shared_csv(subject, "trend_category_span_summary.csv")
tc_df = load_shared_csv(subject, "trend_category_span.csv")

if tc_sum_df is not None and tc_df is not None:
    tab_tc_sum, tab_tc_em, tab_tc_st, tab_tc_dc, tab_tc_hm = st.tabs(
        ["Summary", "📈 Emerging", "🔒 Stable", "📉 Decaying", "Presence Heatmap"]
    )

    with tab_tc_sum:
        tc_f = tc_sum_df[tc_sum_df["model"].isin([MODEL_LABELS[m] for m in selected_models])]
        
        # Overall comparison
        st.subheader("Overall Performance (90 Keywords)")
        tc_all = tc_f[tc_f["category"] == "all"]
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                tc_all, x="model", y="avg_span_paper", color="model", color_discrete_map=colors,
                title="avg-SPAN (Paper Formula)", text_auto=".4f"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                tc_all, x="model", y="n_captured", color="model", color_discrete_map=colors,
                title="Keywords Captured (out of 90)"
            )
            fig.update_layout(height=400, showlegend=False, yaxis_range=[0, 95])
            st.plotly_chart(fig, use_container_width=True)
            
        # By category
        st.subheader("Performance by Category")
        tc_cat = tc_f[tc_f["category"] != "all"]
        fig_cat = px.bar(
            tc_cat, x="category", y="avg_span_paper", color="model", barmode="group",
            color_discrete_map=colors, title="avg-SPAN (Paper) by Trend Category",
            labels={"avg_span_paper": "avg-SPAN (paper)", "category": "Category"}
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
        st.dataframe(
            tc_f[["model", "category", "n_captured", "capture_pct", "avg_span_paper", "avg_span_simple"]].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "model": "Model", "category": "Category", "n_captured": "Captured",
                "capture_pct": st.column_config.NumberColumn("Captured (%)", format="%.1f%%"),
                "avg_span_paper": st.column_config.NumberColumn("avg-SPAN (paper)", format="%.6f"),
                "avg_span_simple": st.column_config.NumberColumn("avg-SPAN (simple)", format="%.2f")
            }
        )

    def render_category_tab(cat_name):
        cat_df = tc_df[tc_df["category"] == cat_name]
        span_cols = ["word", "v_hat"]
        for m in selected_models:
            if f"span_{MODEL_LABELS[m]}" in cat_df.columns:
                span_cols.append(f"span_{MODEL_LABELS[m]}")
                
        st.dataframe(
            cat_df[span_cols].sort_values("v_hat", ascending=False).reset_index(drop=True),
            use_container_width=True,
            column_config={
                "word": "Keyword", "v_hat": "v̂ₖ (corpus freq)",
                **{f"span_{MODEL_LABELS[m]}": f"SPAN ({MODEL_LABELS[m]})" for m in selected_models}
            }
        )

    with tab_tc_em:
        render_category_tab("emerging")
    with tab_tc_st:
        render_category_tab("stable")
    with tab_tc_dc:
        render_category_tab("decaying")

    with tab_tc_hm:
        import ast as _ast
        col_m, col_c = st.columns(2)
        model_hm = col_m.selectbox("Select model for heatmap", selected_models, format_func=lambda x: MODEL_LABELS[x], key="tc_hm_mod")
        cat_hm = col_c.selectbox("Select category", ["emerging", "stable", "decaying"], key="tc_hm_cat")
        
        lbl = MODEL_LABELS[model_hm]
        trend_col = f"trend_{lbl}"
        span_col  = f"span_{lbl}"
        
        hm_df = tc_df[(tc_df["category"] == cat_hm) & (tc_df[span_col] > 0)]
        if len(hm_df) > 0 and trend_col in hm_df.columns:
            rows_hm = []
            for _, r in hm_df.nlargest(30, span_col).iterrows():
                try:
                    trend = _ast.literal_eval(r[trend_col])
                    rows_hm.append({"word": r["word"], "trend": trend, "span": r[span_col]})
                except:
                    pass
            if rows_hm:
                n_years = len(rows_hm[0]["trend"])
                years = list(range(2000, 2000+n_years))
                matrix = [r["trend"] for r in rows_hm]
                words = [f"{r['word']} (S={r['span']})" for r in rows_hm]
                fig = px.imshow(np.array(matrix), x=[str(y) for y in years], y=words, aspect="auto", 
                                color_continuous_scale=[[0, "#1e1e2e"], [1, "#22c55e"]],
                                title=f"Keyword-Trend: {lbl} — {cat_hm.title()} (Max 30 Words)",
                                labels={"color": "Present"})
                fig.update_layout(height=max(400, len(words)*28))
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No heatmap data available.")
else:
    st.info("No Trend Category SPAN data available. Run `trend_category_span.py` first.")

st.divider()

