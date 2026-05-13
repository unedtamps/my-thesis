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
        if col not in display_cols and col not in ["subject", "workers", "random_state", "seed", "passes", "chunksize", "lr_a", "lr_b", "lr_c", "min_cf", "min_df", "ll_per_word", "ngram_range"]:
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

        # ── Parameter Analysis Tabs ───────────────────────────────────────────
        _all_df = None
        _params = []
        
        # 1. Load All Subjects Data & Define Params
        if m == "lda":
            _frames = []
            _base = Path(__file__).resolve().parent.parent.parent / "results" / "lda" / "tuning"
            for _s in ["cs", "math", "physics"]:
                _p = _base / _s / "tuning_results.csv"
                if _p.exists():
                    _t = pd.read_csv(_p)
                    _t["subject"] = _s
                    _frames.append(_t)
            if _frames:
                _all_df = pd.concat(_frames, ignore_index=True)
            _params = ["num_topics", "alpha", "eta", "iterations"]
            
        elif m == "dtm":
            _frames = []
            _base = Path(__file__).resolve().parent.parent.parent / "results" / "dtm" / "tuning"
            for _s in ["cs", "math", "physics"]:
                _p = _base / _s / "tuning_results.csv"
                if _p.exists():
                    _t = pd.read_csv(_p)
                    _t["subject"] = _s
                    _frames.append(_t)
            if _frames:
                _all_df = pd.concat(_frames, ignore_index=True)
                if "k" in _all_df.columns: _all_df = _all_df.rename(columns={"k": "num_topics"})
                if "eta_var" not in _all_df.columns and "eva_var" in _all_df.columns:
                     _all_df = _all_df.rename(columns={"eva_var": "eta_var"})
            _params = ["num_topics", "alpha_var", "phi_var", "train_iter", "eta_var"]
            
        elif m == "topicGpt":
            _p = Path(__file__).resolve().parent.parent.parent / "results" / "topicGpt" / "tunning" / "tuning_results_all.csv"
            if _p.exists():
                _all_df = pd.read_csv(_p)
            _params = ["min_docs"]
            
        elif m == "bertopic":
            _p = Path(__file__).resolve().parent.parent.parent / "results" / "bertopic" / "tuning" / "hdbscan" / "quality_results.csv"
            if _p.exists():
                _all_df = pd.read_csv(_p)
            _params = ["min_cluster_size", "min_samples", "n_neighbors", "n_components"]
            
        # 2. Render Tabs
        if _all_df is not None and len(_all_df) > 0 and _params:
            if "subject" in _all_df.columns:
                _all_df["subject_label"] = _all_df["subject"].map(
                    {"cs": "Computer Science", "math": "Mathematics", "physics": "Physics"}
                )
            else:
                _all_df["subject_label"] = "Unknown"
                
            _subj_colors = {"Computer Science": "#ef4444", "Mathematics": "#f59e0b", "Physics": "#3b82f6"}
            
            st.markdown("---")
            st.markdown(f"#### 📌 {label}: Pengaruh Hyperparameter terhadap Topic Quality")
            st.caption("Gunakan tab di bawah untuk melihat distribusi performa model berdasarkan tiap hyperparameter (menggunakan data semua subject).")
            
            tabs = st.tabs([f"vs {p}" for p in _params])
            for tab, param in zip(tabs, _params):
                with tab:
                    if param not in _all_df.columns:
                        st.warning(f"Parameter '{param}' tidak ditemukan dalam data.")
                        continue
                    
                    # Supaya Plotly tidak menyembunyikan nilai string ("asymmetric", "auto") pada sumbu X yang mayoritas angka,
                    # kita jadikan kolom ini string dan urutkan kategorinya.
                    plot_df = _all_df.copy()
                    plot_df[param] = plot_df[param].astype(str)
                    
                    def custom_sort_key(val):
                        try:
                            return (0, float(val))
                        except ValueError:
                            return (1, val)
                            
                    sorted_cats = sorted(plot_df[param].unique(), key=custom_sort_key)
                    
                    # Box plot (raw distribution)
                    fig_box = px.box(
                        plot_df, x=param, y="topic_quality",
                        color="subject_label",
                        color_discrete_map=_subj_colors,
                        points="all",
                        title=f"Distribusi Topic Quality per {param}",
                        labels={param: param, "topic_quality": "Topic Quality", "subject_label": "Subject"},
                        category_orders={param: sorted_cats}
                    )
                    fig_box.update_layout(
                        height=400,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    col_line, col_box = st.columns(2)
                    
                    with col_line:
                        _agg = plot_df.groupby(["subject_label", param])["topic_quality"].mean().reset_index()
                        _agg[param] = pd.Categorical(_agg[param], categories=sorted_cats, ordered=True)
                        _agg = _agg.sort_values(param)
                        
                        fig_line = go.Figure()
                        for _sl, _grp in _agg.groupby("subject_label"):
                            _c = _subj_colors.get(_sl, "#888")
                            fig_line.add_trace(go.Scatter(
                                x=_grp[param], y=_grp["topic_quality"],
                                mode="lines+markers", name=_sl,
                                line=dict(color=_c, width=2.5),
                                marker=dict(size=8, color=_c, line=dict(width=1, color="white"))
                            ))
                        fig_line.update_layout(
                            title=f"Rata-rata Topic Quality vs {param}",
                            xaxis_title=param, yaxis_title="Mean Topic Quality",
                            height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        fig_line.update_xaxes(type='category', categoryorder='array', categoryarray=sorted_cats)
                        st.plotly_chart(fig_line, use_container_width=True)
                        
                    with col_box:
                        st.plotly_chart(fig_box, use_container_width=True)

        # All configs table
        st.markdown("**All Configs (sorted by Topic Quality)**")
        sorted_df = df.sort_values("topic_quality", ascending=False)
        hide = ["subject", "random_state", "workers", "seed", "passes", "chunksize", "lr_a", "lr_b", "lr_c", "min_cf", "min_df", "ll_per_word", "ngram_range"]
        display_cols = [c for c in sorted_df.columns if c not in hide]
        st.dataframe(sorted_df[display_cols].reset_index(drop=True), use_container_width=True)
