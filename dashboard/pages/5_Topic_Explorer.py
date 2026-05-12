import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import (
    MODELS, MODEL_LABELS, SUBJECTS, SUBJECT_LABELS,
    load_csv, setup_sidebar, model_color_map, RESULTS_DIR
)
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="Topic Explorer", page_icon="🔬", layout="wide")
st.title("🔬 Topic Explorer")
st.markdown("""
Deep-dive into individual topics across all scenarios.
Select a model, search or pick topics, and explore their **evolution**, **consistency**, and **continuity** in one place.

"""
)
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()

# ── Global search and filters ──
col_s, col_f = st.columns([2, 1])
with col_s:
    search = st.text_input(
        "🔍 Search topics by keyword or sentence",
        placeholder="e.g. deep learning for image recognition",
        key="explorer_search",
        help="Enter keywords or a full sentence. At least half the words must match a topic.",
    )
with col_f:
    category_filter = st.multiselect(
        "🏷️ Filter by Category",
        options=["GROWING", "STABLE", "DECLINING"],
        format_func=lambda x: {"GROWING": "📈 Emerging", "STABLE": "🔒 Stable", "DECLINING": "📉 Decaying"}.get(x, x),
        default=[],
        key="explorer_category",
        help="Leave empty to show all."
    )

for m in selected_models:
    label = MODEL_LABELS[m]

    # Load all per-topic data
    prev_df = load_csv(m, "temporal", subject, "topic_prevalence.csv")
    word_df = load_csv(m, "temporal", subject, "topic_word_evolution.csv")
    trends_df = load_csv(m, "temporal", subject, "topic_trends.csv")
    cont_trans = load_csv(m, "continuity", subject, "continuity_transitions.csv")
    labels_df = load_csv(m, "temporal", subject, "topic_labels.csv")
    yearly_desc_df = load_csv(m, "temporal", subject, "topic_yearly_descriptions.csv")

    global_words_path = RESULTS_DIR / m / "modeling" / "global_top_words.csv"
    global_words_df = pd.read_csv(global_words_path) if global_words_path.exists() else None

    # Load trend top words per category for this model + subject
    trend_kw_path = RESULTS_DIR / "shared" / "trend_model" / m / subject / "trend_top_words.csv"
    trend_kw_df = pd.read_csv(trend_kw_path) if trend_kw_path.exists() else None

    # Build label & description lookup from topic_labels.csv
    topic_label_map = {}
    topic_desc_map = {}
    if labels_df is not None and len(labels_df) > 0:
        topic_label_map = dict(zip(labels_df["topic_id"], labels_df["label"]))
        topic_desc_map = dict(zip(labels_df["topic_id"], labels_df["enriched_description"]))

    if prev_df is None or len(prev_df) == 0:
        continue

    with st.expander(f"📊 {label}", expanded=(m == selected_models[0])):
        topic_ids = sorted(prev_df["topic_id"].unique())

        topic_global_words_map = {}
        if global_words_df is not None and len(global_words_df) > 0:
            # Filter by current subject just in case it contains all subjects
            sub_global = global_words_df[global_words_df["subject"] == subject]
            topic_global_words_map = dict(zip(sub_global["topic_id"], sub_global["top_words"]))

        # Build labels + search index from ALL years' words
        t_labels = {}
        t_all_words = {}  # all words across all years for search
        t_word_by_year = {}  # {tid: {year: words}} for hover
        t_categories = {}
        for tid in topic_ids:
            # Use LLM-generated label if available, else fallback
            real_label = topic_label_map.get(tid, "")
            lbl = f"Topic {tid}"
            # Collect words across all years from word evolution
            all_words_set = set()
            year_words = {}
            if word_df is not None and len(word_df) > 0:
                tw = word_df[word_df["topic_id"] == tid]
                for _, wr in tw.iterrows():
                    w = str(wr.get("top_words", ""))
                    year_words[wr["year"]] = w
                    all_words_set.update(w.lower().replace(",", " ").split())

            # Also add label and enriched description to search index
            if real_label:
                all_words_set.update(real_label.lower().replace("-", " ").replace("&", " ").split())
            enrich_desc = str(topic_desc_map.get(tid, "")).lower()
            if enrich_desc and enrich_desc != "nan":
                all_words_set.update(enrich_desc.replace(",", " ").replace(".", " ").split())

            # Also add trend summary words
            trend = ""
            summary_words = ""
            if trends_df is not None and len(trends_df) > 0:
                t_row = trends_df[trends_df["topic_id"] == tid]
                if len(t_row) > 0:
                    trend = t_row.iloc[0].get("trend", "")
                    summary_words = str(t_row.iloc[0].get("top_words", ""))
                    all_words_set.update(summary_words.replace(",", " ").split())
                    icon = {"GROWING": "📈", "DECLINING": "📉", "STABLE": "🔒"}.get(trend, "")
                    if real_label:
                        lbl = f"{icon} T{tid}: {real_label}"
                    else:
                        lbl = f"{icon} T{tid}: {summary_words[:55]}"
            elif real_label:
                lbl = f"T{tid}: {real_label}"

            t_labels[tid] = lbl
            t_all_words[tid] = " ".join(all_words_set).lower()
            t_word_by_year[tid] = year_words
            t_categories[tid] = trend

        # BM25 Search Engine
        bm25_scores = {}
        filtered = topic_ids
        tokens = []
        if search:
            tokens = [tok.strip().lower() for tok in search.replace(",", " ").split() if tok.strip()]
            if tokens:
                tokenized_corpus = [t_all_words[t].split() for t in topic_ids]
                bm25 = BM25Okapi(tokenized_corpus)
                doc_scores = bm25.get_scores(tokens)
                
                # Keep topics with score > 0
                scored_topics = [(t, score) for t, score in zip(topic_ids, doc_scores) if score > 0]
                scored_topics.sort(key=lambda x: x[1], reverse=True)
                
                # Batasi hanya menampilkan top 15 topik yang paling relevan
                scored_topics = scored_topics[:15]
                
                filtered = [t for t, _ in scored_topics]
                bm25_scores = {t: score for t, score in scored_topics}

        if category_filter:
            filtered = [t for t in filtered if t_categories.get(t, "") in category_filter]
            
        if not filtered:
            if search or category_filter:
                st.caption(f"No topics match your filters in {label}")
            continue
        
        if search or category_filter:
            st.caption(f"{len(filtered)} topics match your filters")

        # Compute total article count per topic (sum doc_count across years)
        t_doc_counts = {}
        if prev_df is not None and "doc_count" in prev_df.columns:
            for tid in topic_ids:
                t_doc_counts[tid] = int(prev_df[prev_df["topic_id"] == tid]["doc_count"].sum())

        # Compute R² per topic from trends
        t_r_squared = {}
        if trends_df is not None and len(trends_df) > 0 and "r_squared" in trends_df.columns:
            for tid in topic_ids:
                t_row = trends_df[trends_df["topic_id"] == tid]
                if len(t_row) > 0:
                    t_r_squared[tid] = float(t_row.iloc[0]["r_squared"])

        # ── Category Distribution Summary ──
        all_cats = [t_categories.get(t, "") for t in topic_ids if t_categories.get(t, "") != ""]
        total_cats = len(all_cats)
        cat_order = ["GROWING", "STABLE", "DECLINING"]
        cat_icons = {"GROWING": "📈", "STABLE": "🔒", "DECLINING": "📉"}
        cat_colors_hex = {"GROWING": "#22c55e", "STABLE": "#3b82f6", "DECLINING": "#ef4444"}
        cat_bg = {"GROWING": "#052e16", "STABLE": "#1e3a5f", "DECLINING": "#450a0a"}

        if total_cats > 0:
            st.markdown("**📊 Distribusi Kategori Topik**")
            cat_metric_cols = st.columns(3)
            for ci, cat in enumerate(cat_order):
                count = all_cats.count(cat)
                pct = count / total_cats * 100
                icon = cat_icons[cat]
                color = cat_colors_hex[cat]
                bg = cat_bg[cat]
                with cat_metric_cols[ci]:
                    st.markdown(
                        f"""
                        <div style='
                            background:{bg};
                            border:1px solid {color}44;
                            border-left: 4px solid {color};
                            border-radius:8px;
                            padding:10px 14px;
                            margin-bottom:8px;
                        '>
                            <div style='font-size:0.78em;color:{color};font-weight:600;letter-spacing:0.05em'>{icon} {cat}</div>
                            <div style='font-size:1.7em;font-weight:700;color:#fff;line-height:1.2'>{count}</div>
                            <div style='font-size:0.88em;color:#aaa'>{pct:.1f}% dari {total_cats} topik</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

        # Sort option
        sort_options = ["Topic ID", "📊 Article Count (↓)", "📐 R² (↓)"]
        if search and len(tokens) > 0:
            sort_options.insert(0, "⭐ Relevance Score (↓)")
            
        sort_by = st.radio(
            "Sort by", sort_options,
            horizontal=True, key=f"sort_{m}",
        )
        
        if sort_by == "⭐ Relevance Score (↓)":
            filtered = sorted(filtered, key=lambda t: bm25_scores.get(t, 0), reverse=True)
        elif sort_by == "📊 Article Count (↓)":
            filtered = sorted(filtered, key=lambda t: t_doc_counts.get(t, 0), reverse=True)
        elif sort_by == "📐 R² (↓)":
            filtered = sorted(filtered, key=lambda t: t_r_squared.get(t, 0), reverse=True)

        # Append article count and R² to labels for display
        t_labels_with_count = {}
        for tid in filtered:
            cnt = t_doc_counts.get(tid, 0)
            r2 = t_r_squared.get(tid, None)
            r2_str = f" | R²={r2:.3f}" if r2 is not None else ""
            score_str = f" | ⭐ {bm25_scores.get(tid, 0):.2f}" if search and bm25_scores.get(tid, 0) > 0 else ""
            t_labels_with_count[tid] = f"{t_labels.get(tid, f'Topic {tid}')}  [{cnt} articles{r2_str}{score_str}]"

        picks = st.multiselect(
            "Topics", filtered,
            default=filtered[:2] if len(filtered) >= 2 else filtered[:1],
            format_func=lambda x: t_labels_with_count.get(x, t_labels.get(x, f"Topic {x}")),
            key=f"explorer_{m}",
        )

        if not picks:
            st.info("Select at least one topic.")
            continue

        # ────────────────────────────────────────────
        # TAB LAYOUT per model
        # ────────────────────────────────────────────
        t_evo, t_cont, t_kw = st.tabs([
            "📈 Evolution & Prevalence",
            "🔄 Continuity",
            "🏷️ Trend Keywords",
        ])

        # ── TAB 1: Evolution ──
        with t_evo:
            # Prevalence chart with per-year word hover
            fig = go.Figure()
            for tid in picks:
                tp = prev_df[prev_df["topic_id"] == tid].sort_values("year")
                if len(tp) > 0:
                    # Build hover text with year-specific words
                    hover_words = []
                    for _, row in tp.iterrows():
                        yr = row["year"]
                        w = t_word_by_year.get(tid, {}).get(yr, "")
                        if len(w) > 80:
                            w = w[:80] + "..."
                        hover_words.append(w)

                    fig.add_trace(go.Scatter(
                        x=tp["year"], y=tp["proportion"],
                        name=f"T{tid}", mode="lines+markers",
                        line=dict(width=2), marker=dict(size=4),
                        customdata=list(zip(hover_words, tp["doc_count"].tolist())),
                        hovertemplate=(
                            "<b>T%{x}</b> — %{y:.4f}<br>"
                            "Docs: %{customdata[1]}<br>"
                            "Words: %{customdata[0]}"
                            "<extra>T" + str(tid) + "</extra>"
                        ),
                    ))
            fig.update_layout(
                title=f"{label} — Topic Prevalence",
                xaxis_title="Year", yaxis_title="Proportion",
                height=380, hovermode="x",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Per-topic details
            for tid in picks:
                trend_tag = ""
                if trends_df is not None and len(trends_df) > 0:
                    t_row = trends_df[trends_df["topic_id"] == tid]
                    if len(t_row) > 0:
                        trend_tag = t_row.iloc[0].get("trend", "")
                icon = {"GROWING": "📈", "DECLINING": "📉", "STABLE": "🔒"}.get(trend_tag, "")
                evo_label = topic_label_map.get(tid, f"Topic {tid}")

                with st.expander(f"{icon} Topic {tid}: {evo_label} — evolution details"):
                    # Global top words
                    g_words = topic_global_words_map.get(tid, "")
                    if g_words and str(g_words) != "nan":
                        st.markdown(f"**Global Top Words:** {g_words}")
                        
                    # Enriched description
                    tid_desc = topic_desc_map.get(tid, "")
                    if tid_desc and str(tid_desc) != "nan":
                        st.info(tid_desc)

                    # Stats
                    if trends_df is not None and len(trends_df) > 0:
                        t_row = trends_df[trends_df["topic_id"] == tid]
                        if len(t_row) > 0:
                            tr = t_row.iloc[0]
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Trend", f"{icon} {tr.get('trend', 'N/A')}")
                            c2.metric("Slope", f"{tr.get('slope', 0):.6f}")
                            c3.metric("R²", f"{tr.get('r_squared', 0):.4f}")

                    # Word evolution + yearly descriptions merged
                    if word_df is not None and len(word_df) > 0:
                        tw = word_df[word_df["topic_id"] == tid].sort_values("year")
                        if len(tw) > 0:
                            display = tw[["year", "top_words"]].copy()
                            display.columns = ["Year", "Top Words"]
                            # Merge yearly descriptions if available
                            if yearly_desc_df is not None and len(yearly_desc_df) > 0:
                                yd = yearly_desc_df[yearly_desc_df["topic_id"] == tid][["year", "yearly_description"]].copy()
                                yd.columns = ["Year", "Description"]
                                display = display.merge(yd, on="Year", how="left")
                                display["Description"] = display["Description"].fillna("")
                            st.dataframe(display.reset_index(drop=True), use_container_width=True, height=300)


        # ── TAB 3: Continuity ──
        with t_cont:
            if cont_trans is not None and len(cont_trans) > 0:
                ct_data = cont_trans[cont_trans["topic_id"].isin(picks)]
                if len(ct_data) > 0:
                    # Category colors
                    cat_colors = {
                        "stable": "#22c55e", "merged": "#f59e0b",
                        "merge": "#f59e0b",
                        "disappear": "#ef4444", "new": "#3b82f6",
                    }

                    # Continuity timeline per topic
                    for tid in picks:
                        tc = ct_data[ct_data["topic_id"] == tid].sort_values("year_from")
                        if len(tc) == 0:
                            continue

                        st.markdown(f"**Topic {tid} — Continuity Timeline**")

                        # Stacked category chart
                        cat_counts = tc["category"].value_counts().to_dict()
                        c_cols = st.columns(len(cat_counts) if cat_counts else 1)
                        for i, (cat, cnt) in enumerate(cat_counts.items()):
                            with c_cols[i]:
                                st.metric(cat.capitalize(), cnt)

                        # Timeline
                        fig = go.Figure()
                        for cat in tc["category"].unique():
                            cat_data = tc[tc["category"] == cat]
                            fig.add_trace(go.Scatter(
                                x=cat_data["year_from"],
                                y=[f"T{tid}"] * len(cat_data),
                                mode="markers",
                                name=cat.capitalize(),
                                marker=dict(
                                    size=12,
                                    color=cat_colors.get(cat, "#888"),
                                    symbol="square",
                                ),
                                hovertemplate=(
                                    "Year: %{x}<br>"
                                    f"Category: {cat}<br>"
                                    "Best match: %{customdata[0]}<br>"
                                    "Sim: %{customdata[1]:.3f}"
                                ),
                                customdata=cat_data[["best_match_topic", "best_match_sim"]].values
                                if "best_match_topic" in cat_data.columns else None,
                            ))
                        fig.update_layout(
                            height=120, showlegend=True,
                            margin=dict(l=0, r=0, t=10, b=0),
                            legend=dict(orientation="h", y=-0.5),
                            xaxis_title="Year",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Transition details
                        with st.expander(f"📋 Topic {tid} — transition details"):
                            show_cols = ["year_from", "year_to", "category", "best_match_topic", "best_match_sim"]
                            available = [c for c in show_cols if c in tc.columns]
                            if "words" in tc.columns:
                                available.append("words")
                            st.dataframe(tc[available].reset_index(drop=True), use_container_width=True, height=250)
                else:
                    st.info("No continuity data for selected topics.")
            else:
                st.info("No continuity data available.")

        # ── TAB 4: Trend Keywords ──
        with t_kw:
            st.markdown(
                "Top-20 kata dengan frekuensi kemunculan tertinggi dari topik-topik "
                "yang dikategorikan **GROWING**, **STABLE**, dan **DECLINING** "
                f"dalam model **{label}** untuk subjek **{subject.upper()}**."
            )
            if trend_kw_df is None or len(trend_kw_df) == 0:
                st.info("Trend keyword data not available for this model / subject.")
            else:
                TREND_ICON = {"GROWING": "📈", "STABLE": "🔒", "DECLINING": "📉"}
                TREND_COLOR = {
                    "GROWING":  "#22c55e",  # green
                    "STABLE":   "#3b82f6",  # blue
                    "DECLINING": "#ef4444", # red
                }
                cat_cols = st.columns(3)

                for col_idx, cat in enumerate(["GROWING", "STABLE", "DECLINING"]):
                    cat_df = trend_kw_df[trend_kw_df["trend"] == cat].sort_values("rank")
                    icon = TREND_ICON[cat]
                    color = TREND_COLOR[cat]
                    n_topics = int(cat_df.iloc[0]["n_topics_in_category"]) if len(cat_df) > 0 else 0

                    with cat_cols[col_idx]:
                        st.markdown(
                            f"<h5 style='color:{color}; margin-bottom:4px'>"
                            f"{icon} {cat} <span style='font-size:0.75em; color:#888'>({n_topics} topics)</span>"
                            f"</h5>",
                            unsafe_allow_html=True,
                        )
                        if cat_df.empty:
                            st.caption("No data")
                        else:
                            # Display as a styled table
                            rows_html = "".join(
                                f"<tr>"
                                f"<td style='color:#888;font-size:0.78em;padding:2px 6px'>{int(r['rank'])}</td>"
                                f"<td style='font-weight:500;padding:2px 6px'>{r['word']}</td>"
                                f"<td style='color:#888;font-size:0.78em;text-align:right;padding:2px 6px'>{int(r['freq'])}</td>"
                                f"</tr>"
                                for _, r in cat_df.iterrows()
                            )
                            st.markdown(
                                f"<table style='width:100%;border-collapse:collapse;font-size:0.88em'>"
                                f"<thead><tr>"
                                f"<th style='text-align:left;color:#aaa;font-size:0.78em;padding:2px 6px'>#</th>"
                                f"<th style='text-align:left;color:#aaa;font-size:0.78em;padding:2px 6px'>Word</th>"
                                f"<th style='text-align:right;color:#aaa;font-size:0.78em;padding:2px 6px'>Freq</th>"
                                f"</tr></thead>"
                                f"<tbody>{rows_html}</tbody>"
                                f"</table>",
                                unsafe_allow_html=True,
                            )
