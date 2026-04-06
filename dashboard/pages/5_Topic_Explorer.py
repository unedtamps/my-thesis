import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import (
    MODELS, MODEL_LABELS, SUBJECTS, SUBJECT_LABELS,
    load_csv, setup_sidebar, model_color_map,
)

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
    ttd_yoy = load_csv(m, "consistency", subject, "ttd_yoy_per_topic.csv")
    ttd_endpoint = load_csv(m, "consistency", subject, "topic_term_drift.csv")
    ttd_traj = load_csv(m, "consistency", subject, "ttd_trajectory.csv")
    cont_trans = load_csv(m, "consistency", subject, "continuity_transitions.csv")
    labels_df = load_csv(m, "temporal", subject, "topic_labels.csv")
    yearly_desc_df = load_csv(m, "temporal", subject, "topic_yearly_descriptions.csv")

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

        # Filter by global search (threshold-based: at least half the tokens must match)
        filtered = topic_ids
        if search:
            tokens = [tok.strip().lower() for tok in search.replace(",", " ").split() if tok.strip()]
            if tokens:
                min_matches = max(1, len(tokens) // 2)
                filtered = [
                    t for t in filtered
                    if sum(1 for tok in tokens if tok in t_all_words[t]) >= min_matches
                ]
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

        # Sort option
        sort_by = st.radio(
            "Sort by", ["Topic ID", "📊 Article Count (↓)", "📐 R² (↓)"],
            horizontal=True, key=f"sort_{m}",
        )
        if sort_by == "📊 Article Count (↓)":
            filtered = sorted(filtered, key=lambda t: t_doc_counts.get(t, 0), reverse=True)
        elif sort_by == "📐 R² (↓)":
            filtered = sorted(filtered, key=lambda t: t_r_squared.get(t, 0), reverse=True)

        # Append article count and R² to labels for display
        t_labels_with_count = {}
        for tid in filtered:
            cnt = t_doc_counts.get(tid, 0)
            r2 = t_r_squared.get(tid, None)
            r2_str = f" | R²={r2:.3f}" if r2 is not None else ""
            t_labels_with_count[tid] = f"{t_labels.get(tid, f'Topic {tid}')}  [{cnt} articles{r2_str}]"

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
        t_evo, t_consist, t_cont = st.tabs([
            "📈 Evolution & Prevalence",
            "🔗 Term Drift (TTD)",
            "🔄 Continuity",
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

        # ── TAB 2: Term Drift ──
        with t_consist:
            # Trajectory chart: drift from baseline per topic
            if ttd_traj is not None and len(ttd_traj) > 0:
                traj_data = ttd_traj[ttd_traj["topic_id"].isin(picks)]
                if len(traj_data) > 0:
                    fig = go.Figure()
                    for tid in picks:
                        td = traj_data[traj_data["topic_id"] == tid].sort_values("year")
                        if len(td) > 0:
                            fig.add_trace(go.Scatter(
                                x=td["year"], y=td["drift_from_baseline"],
                                name=f"T{tid}", mode="lines+markers",
                                line=dict(width=2), marker=dict(size=4),
                            ))
                    fig.update_layout(
                        title=f"{label} — Vocabulary Drift from Baseline",
                        xaxis_title="Year", yaxis_title="Drift (TTD)",
                        height=380, hovermode="x unified",
                        legend=dict(orientation="h", y=-0.15),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trajectory data for selected topics.")
            else:
                st.info("No TTD trajectory data available.")

            # Endpoint + YoY summary per topic
            for tid in picks:
                with st.expander(f"📋 Topic {tid} — drift details"):
                    c1, c2 = st.columns(2)

                    # Endpoint drift
                    with c1:
                        st.markdown("**Endpoint Drift**")
                        if ttd_endpoint is not None and len(ttd_endpoint) > 0:
                            ep = ttd_endpoint[ttd_endpoint["topic_id"] == tid]
                            if len(ep) > 0:
                                row = ep.iloc[0]
                                st.metric("TTD (first → last)", f"{row.get('ttd', 0):.4f}")
                                st.metric("Cosine Sim", f"{row.get('cosine_sim', 0):.4f}")
                                st.caption(f"**First words:** {row.get('words_first', '')[:80]}")
                                st.caption(f"**Last words:** {row.get('words_last', '')[:80]}")
                            else:
                                st.caption("No endpoint data")
                        else:
                            st.caption("No endpoint data")

                    # YoY summary
                    with c2:
                        st.markdown("**Year-over-Year**")
                        if ttd_yoy is not None and len(ttd_yoy) > 0:
                            yoy = ttd_yoy[ttd_yoy["topic_id"] == tid]
                            if len(yoy) > 0:
                                row = yoy.iloc[0]
                                st.metric("Avg YoY Drift", f"{row.get('avg_yoy_drift', 0):.4f}")
                                st.metric("Max Single Jump", f"{row.get('max_single_jump', 0):.4f}")
                                avg_sim = row.get("avg_yoy_sim", 0)
                                st.metric("Avg YoY Sim", f"{avg_sim:.4f}")
                            else:
                                st.caption("No YoY data")
                        else:
                            st.caption("No YoY data")

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
