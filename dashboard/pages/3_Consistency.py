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

st.set_page_config(page_title="Scenario 3: Consistency", page_icon="🔗", layout="wide")
st.title("🔗 Scenario 3: Structural Consistency")
st.markdown("""
Evaluates how structurally stable topics remain over time.

**Topic Term Drift (TTD)** — measures vocabulary change using 5-year windows (2000–2005, 2005–2010, ..., 2020–2025):
- *Sliding Window Drift*: RBO drift between consecutive 5-year windows
- *Endpoint Drift*: RBO drift from window 2000–2005 → 2020–2025 (using top-60 words)

**Continuity Rate** — classifies topic transitions (RBO similarity-based):
- *Stable*: sim ≥ 0.7, maps 1:1 to next year
- *Evolve*: 0.3 ≤ sim < 0.7, maps 1:1 (topic shifted but recognisable)
- *Merge*: sim ≥ 0.3, but multiple topics → one
- *Disappear*: sim < 0.3, no match found
- *New*: topic appears without predecessor
""")
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()

WINDOWS = ["2000–2005", "2005–2010", "2010–2015", "2015–2020", "2020–2025"]
TRANSITIONS = [
    "2000–2005 → 2005–2010",
    "2005–2010 → 2010–2015",
    "2010–2015 → 2015–2020",
    "2015–2020 → 2020–2025",
]

# ============================================================
# Topic Term Drift — Sliding Window
# ============================================================
st.header("Topic Term Drift (TTD)")
st.divider()

tab_sliding, tab_endpoint = st.tabs(["📊 Sliding Window Drift (5-Year)", "📍 Endpoint Drift (2000–2005 → 2020–2025)"])

# ── Tab 1: Sliding Window ────────────────────────────────────────────────────
with tab_sliding:
    st.markdown("""
    Average RBO drift between consecutive 5-year windows across all topics.
    Words per window are aggregated by frequency across all years in that window.
    """)

    avg_data = load_all_models("consistency", subject, "window_drift_avg.csv")

    if len(avg_data) > 0:
        avg_f = avg_data[avg_data["model"].isin([MODEL_LABELS[m] for m in selected_models])].copy()

        # Sort transitions in order
        transition_order = {t: i for i, t in enumerate(TRANSITIONS)}
        avg_f["t_order"] = avg_f["transition"].map(transition_order)
        avg_f = avg_f.sort_values("t_order")

        fig = px.line(
            avg_f,
            x="transition", y="avg_drift",
            color="model",
            color_discrete_map=colors,
            markers=True,
            error_y="std_drift",
            title=f"Average Sliding Window Drift — {SUBJECT_LABELS[subject]}",
            labels={"avg_drift": "Avg Drift (1 − RBO)", "transition": "Window Transition"},
        )
        fig.update_layout(
            height=420,
            xaxis_tickangle=-20,
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(range=[0, 1]),
        )
        fig.add_hline(y=0.4, line_dash="dot", line_color="gray",
                      annotation_text="Low drift", annotation_position="right")
        fig.add_hline(y=0.7, line_dash="dot", line_color="orange",
                      annotation_text="High drift", annotation_position="right")
        st.plotly_chart(fig, use_container_width=True)

        # ── Per-transition bar comparison ────────────────────────────────
        st.subheader("Per-Transition Comparison Across Models")
        fig2 = px.bar(
            avg_f,
            x="transition", y="avg_drift",
            color="model",
            barmode="group",
            color_discrete_map=colors,
            error_y="std_drift",
            title=f"Drift per Window Transition — {SUBJECT_LABELS[subject]}",
            labels={"avg_drift": "Avg Drift", "transition": "Window Transition"},
        )
        fig2.update_layout(
            height=380,
            xaxis_tickangle=-20,
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Summary table ────────────────────────────────────────────────
        st.subheader("Overall Summary")
        sum_rows = []
        for m in selected_models:
            df = load_csv(m, "consistency", subject, "window_drift_summary.csv")
            if df is not None and len(df) > 0:
                row = df.iloc[0].to_dict()
                row["model"] = MODEL_LABELS[m]
                sum_rows.append(row)

        if sum_rows:
            sdf = pd.DataFrame(sum_rows).set_index("model")
            display_cols = [c for c in sdf.columns if c not in ("subject",)]
            fmt = {c: "{:.4f}" for c in display_cols if sdf[c].dtype in ("float64", "float32")}
            st.dataframe(sdf[display_cols].style.format(fmt, na_rep="—"), use_container_width=True)
    else:
        st.info("No sliding window drift data found. Run the consistency notebooks first.")

    # ── Per-topic detail for single model ───────────────────────────────
    if len(selected_models) == 1:
        m = selected_models[0]
        st.subheader(f"Per-Topic Drift Detail — {MODEL_LABELS[m]}")
        wdf = load_csv(m, "consistency", subject, "window_drift.csv")
        if wdf is not None and len(wdf) > 0:
            topic_avg = wdf.groupby("topic_id")["drift"].mean().reset_index()
            topic_avg.columns = ["topic_id", "avg_drift"]
            topic_avg = topic_avg.sort_values("avg_drift", ascending=False)

            fig3 = px.bar(
                topic_avg.head(30),
                x="topic_id", y="avg_drift",
                title=f"Top 30 Most Drifted Topics — {MODEL_LABELS[m]}",
                labels={"topic_id": "Topic ID", "avg_drift": "Avg Drift across windows"},
                color="avg_drift",
                color_continuous_scale="Reds",
            )
            fig3.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

            with st.expander("📄 Full window drift table"):
                show_cols = ["topic_id", "window_from", "window_to", "rbo_sim", "drift"]
                available = [c for c in show_cols if c in wdf.columns]
                st.dataframe(wdf[available].sort_values(["topic_id", "window_from"]),
                             use_container_width=True, height=400)

    # ── Topic Example Table (HTML merged cells) ─────────────────────────
    st.subheader("🔍 Topic Example: Window Words & Drift (Topic 0 and Topic 1)")
    st.caption(
        "Row 1 = Topic ID (merged across all windows) · "
        "Row 2 = Drift to next window · "
        "Row 3 = Top-10 words for that window"
    )

    _ex_model = selected_models[0] if selected_models else None
    if _ex_model:
        _wt = load_csv(_ex_model, "consistency", subject, "window_topwords.csv")
        _wd = load_csv(_ex_model, "consistency", subject, "window_drift.csv")

        if _wt is not None and len(_wt) > 0:
            EXAMPLE_TOPICS = [0, 1]
            _wins = WINDOWS  # 5 windows
            # drift[i] = transition from _wins[i] → _wins[i+1]
            _trans_for_win = {}  # win_label → transition string or None (last window)
            for i, w in enumerate(_wins):
                _trans_for_win[w] = TRANSITIONS[i] if i < len(TRANSITIONS) else None

            # Look-up dicts
            win_words = {}   # (topic_id, window) → top_words string
            drift_map = {}   # (topic_id, transition) → drift float

            if _wt is not None:
                for _, r in _wt.iterrows():
                    win_words[(int(r["topic_id"]), r["window"])] = \
                        str(r.get("top_words", "")) if pd.notna(r.get("top_words", "")) else ""

            if _wd is not None:
                for _, r in _wd.iterrows():
                    drift_map[(int(r["topic_id"]), r["transition"])] = float(r.get("drift", 0))

            # ── Build HTML table ──────────────────────────────────────────
            n_cols = 1 + len(_wins)   # label col + 5 window cols

            css = """
            <style>
            .ttd { border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 13px; }
            .ttd th  { background:#1e293b; color:#f8fafc; padding:8px 10px; text-align:center; border:1px solid #334155; }
            .ttd td  { border:1px solid #cbd5e1; padding:7px 10px; vertical-align:top; }
            .ttd .lbl { background:#f1f5f9; font-weight:600; color:#475569; white-space:nowrap; width:90px; }
            .ttd .tid { background:#312e81; color:#e0e7ff; font-weight:700;
                        text-align:center; font-size:14px; letter-spacing:.5px; }
            .ttd .ds  { background:#dcfce7; color:#14532d; text-align:center; font-weight:600; }
            .ttd .dm  { background:#fef9c3; color:#713f12; text-align:center; font-weight:600; }
            .ttd .dh  { background:#fee2e2; color:#7f1d1d; text-align:center; font-weight:600; }
            .ttd .dn  { color:#94a3b8;      text-align:center; }
            .ttd .wc  { font-size:12px; color:#1e293b; line-height:1.5; }
            </style>
            """

            html = css + '<table class="ttd"><thead><tr><th></th>'
            for w in _wins:
                html += f"<th>{w}</th>"
            html += "</tr></thead><tbody>"

            for tid in EXAMPLE_TOPICS:
                # Row 1 — Topic ID (merged)
                html += (
                    f'<tr><td class="tid" colspan="{n_cols}">'
                    f'Topic {tid} &nbsp;—&nbsp; {MODEL_LABELS[_ex_model]}'
                    f'</td></tr>'
                )

                # Row 2 — Drift (drift to NEXT window; last window shows —)
                html += '<tr><td class="lbl">Drift →</td>'
                for w in _wins:
                    trans = _trans_for_win[w]
                    if trans is None:
                        html += '<td class="dn">—</td>'
                    else:
                        d = drift_map.get((tid, trans), None)
                        if d is None:
                            html += '<td class="dn">—</td>'
                        elif d < 0.4:
                            html += f'<td class="ds">{d:.4f}</td>'
                        elif d < 0.7:
                            html += f'<td class="dm">{d:.4f}</td>'
                        else:
                            html += f'<td class="dh">{d:.4f}</td>'
                html += "</tr>"

                # Row 3 — Top-10 words
                html += '<tr><td class="lbl">Top Words</td>'
                for w in _wins:
                    raw = win_words.get((tid, w), "")
                    if raw:
                        top10 = ", ".join(x.strip() for x in raw.split(",")[:10])
                        html += f'<td class="wc">{top10}</td>'
                    else:
                        html += '<td class="dn">—</td>'
                html += "</tr>"

            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
            st.caption("🟢 Drift < 0.4 Stable · 🟡 0.4–0.7 Moderate · 🔴 ≥ 0.7 High · — = not active / last window")
        else:
            st.info(f"No window top-words data for {MODEL_LABELS[_ex_model]}.")

# ── Tab 2: Endpoint Drift ────────────────────────────────────────────────────
with tab_endpoint:
    st.markdown("""
    Long-term drift from **2000–2005** → **2020–2025**, using top-60 aggregated words per window.
    This captures total vocabulary change over the full 25-year span.
    """)

    # ── Summary metrics across models ────────────────────────────────────
    ep_sum_rows = []
    for m in selected_models:
        df = load_csv(m, "consistency", subject, "endpoint_drift_summary.csv")
        if df is not None and len(df) > 0:
            row = df.iloc[0].to_dict()
            row["model"] = MODEL_LABELS[m]
            ep_sum_rows.append(row)

    if ep_sum_rows:
        ep_sum_df = pd.DataFrame(ep_sum_rows)

        # Metric row
        col_items = list(zip(ep_sum_df["model"], ep_sum_df.get("avg_drift", [None]*len(ep_sum_df)),
                             ep_sum_df.get("avg_pct_kept", [None]*len(ep_sum_df))))
        cols = st.columns(len(col_items))
        for col, (mdl, drift, kept) in zip(cols, col_items):
            with col:
                st.metric(
                    label=f"🔵 {mdl}",
                    value=f"{drift:.4f}" if drift is not None else "—",
                    delta=f"{kept:.1f}% vocab kept" if kept is not None else None,
                    delta_color="off",
                )

        # Bar chart: avg drift comparison
        fig = px.bar(
            ep_sum_df.sort_values("avg_drift", ascending=False)
            if "avg_drift" in ep_sum_df.columns else ep_sum_df,
            x="model", y="avg_drift",
            color="model",
            color_discrete_map=colors,
            error_y="std_drift" if "std_drift" in ep_sum_df.columns else None,
            title=f"Endpoint Drift (2000–2005 → 2020–2025) — {SUBJECT_LABELS[subject]}",
            labels={"avg_drift": "Avg Endpoint Drift (1 − RBO)", "model": "Model"},
        )
        fig.update_layout(height=380, showlegend=False, yaxis=dict(range=[0, 1]))
        fig.add_hline(y=0.7, line_dash="dot", line_color="orange",
                      annotation_text="High drift", annotation_position="right")
        st.plotly_chart(fig, use_container_width=True)

        # Distribution: stable / moderate / high
        dist_cols = ["pct_stable", "pct_moderate", "pct_high_drift"]
        available_dist = [c for c in dist_cols if c in ep_sum_df.columns]
        if available_dist:
            dist_melt = ep_sum_df[["model"] + available_dist].melt(
                id_vars="model", var_name="Category", value_name="% Topics"
            )
            dist_melt["Category"] = dist_melt["Category"].map({
                "pct_stable": "Stable (<0.4)",
                "pct_moderate": "Moderate (0.4–0.7)",
                "pct_high_drift": "High Drift (≥0.7)",
            })
            fig2 = px.bar(
                dist_melt, x="model", y="% Topics", color="Category",
                barmode="stack",
                color_discrete_map={
                    "Stable (<0.4)": "#22c55e",
                    "Moderate (0.4–0.7)": "#f59e0b",
                    "High Drift (≥0.7)": "#ef4444",
                },
                title=f"Endpoint Drift Distribution by Model — {SUBJECT_LABELS[subject]}",
            )
            fig2.update_layout(height=380, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig2, use_container_width=True)

        # Full summary table
        with st.expander("📄 Full summary table"):
            display_cols = [c for c in ep_sum_df.columns if c not in ("subject",)]
            fmt = {c: "{:.4f}" for c in display_cols
                   if c != "model" and ep_sum_df[c].dtype in ("float64", "float32")}
            st.dataframe(
                ep_sum_df[display_cols].set_index("model").style.format(fmt, na_rep="—"),
                use_container_width=True,
            )
    else:
        st.info("No endpoint drift data found.")

    # ── Per-topic detail for single model ───────────────────────────────
    if len(selected_models) == 1:
        m = selected_models[0]
        st.subheader(f"Per-Topic Endpoint Drift — {MODEL_LABELS[m]}")
        ep_df = load_csv(m, "consistency", subject, "endpoint_drift.csv")
        if ep_df is not None and len(ep_df) > 0:
            ep_sorted = ep_df.sort_values("endpoint_drift", ascending=False)

            fig3 = px.bar(
                ep_sorted.head(30),
                x="topic_id", y="endpoint_drift",
                color="endpoint_drift",
                color_continuous_scale="RdYlGn_r",
                title=f"Top 30 Most Drifted Topics (Endpoint) — {MODEL_LABELS[m]}",
                labels={"topic_id": "Topic ID", "endpoint_drift": "Endpoint Drift"},
            )
            fig3.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

            # Scatter: drift vs pct_kept
            if "pct_kept" in ep_df.columns:
                fig4 = px.scatter(
                    ep_df,
                    x="pct_kept", y="endpoint_drift",
                    hover_data=["topic_id", "words_kept"] if "words_kept" in ep_df.columns else ["topic_id"],
                    title=f"Endpoint Drift vs. Vocabulary Retained — {MODEL_LABELS[m]}",
                    labels={"pct_kept": "% Vocabulary Kept (2000–2005 words still in 2020–2025)",
                            "endpoint_drift": "Endpoint Drift"},
                    color="endpoint_drift",
                    color_continuous_scale="RdYlGn_r",
                )
                fig4.update_layout(height=380)
                st.plotly_chart(fig4, use_container_width=True)

            with st.expander("📄 Full endpoint drift table"):
                show_cols = ["topic_id", "rbo_sim", "endpoint_drift", "n_kept", "n_added",
                             "n_removed", "pct_kept", "words_kept"]
                avail = [c for c in show_cols if c in ep_df.columns]
                st.dataframe(ep_sorted[avail].reset_index(drop=True),
                             use_container_width=True, height=400)

# ============================================================
# Continuity Rate
# ============================================================
st.header("Continuity Rate")
st.divider()

tab_timeline, tab_compare = st.tabs(["Timeline", "Model Comparison"])

with tab_timeline:
    for m in selected_models:
        df = load_csv(m, "consistency", subject, "continuity_summary.csv")
        if df is None or len(df) == 0:
            continue

        with st.expander(f"📋 {MODEL_LABELS[m]} — Continuity Timeline", expanded=len(selected_models) == 1):
            df["transition"] = df["year_from"].astype(str) + "→" + df["year_to"].astype(str)

            fig = go.Figure()
            _cat_colors = [
                ("pct_stable",   "#22c55e",  "Stable"),
                ("pct_evolve",   "#3b82f6",  "Evolve"),
                ("pct_merge",    "#f59e0b",  "Merge"),
                ("pct_disappear","#ef4444",  "Disappear"),
            ]
            for cat, color, label in _cat_colors:
                if cat in df.columns:
                    fig.add_trace(go.Bar(
                        x=df["transition"], y=df[cat],
                        name=label,
                        marker_color=color,
                    ))

            fig.update_layout(
                barmode="stack", height=380,
                title=f"{MODEL_LABELS[m]}: Continuity Rate Over Time",
                yaxis_title="Percentage (%)",
                yaxis=dict(range=[0, 100]),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

            if "n_new" in df.columns and "n_topics_t1" in df.columns:
                first_year = df["year_from"].min()
                years = [str(first_year)] + df["year_to"].astype(str).tolist()
                pcts = [100.0] + (df["n_new"] / df["n_topics_t1"] * 100).tolist()

                df_new = pd.DataFrame({"year": years, "pct_new": pcts})
                fig_new = px.line(
                    df_new, x="year", y="pct_new", markers=True,
                    title=f"{MODEL_LABELS[m]}: Percentage of New Topics per Year"
                )
                fig_new.update_traces(line_color="#3b82f6")
                fig_new.update_layout(height=350, yaxis_title="Percentage (%)", xaxis_title="Year")
                st.plotly_chart(fig_new, use_container_width=True)

            st.markdown("##### 🔍 Topic Example: Transition details (Topic 0 & 1)")
            trans_df = load_csv(m, "consistency", subject, "continuity_transitions.csv")
            if trans_df is not None and not trans_df.empty:
                ex_df = trans_df[trans_df["topic_id"].isin([0, 1])].copy()
                if not ex_df.empty:
                    ex_df["transition"] = ex_df["year_from"].astype(str) + " → " + ex_df["year_to"].astype(str)
                    
                    # Highlight colors for category
                    def _style_category(val):
                        if val == "stable":   return "color: #166534; font-weight: bold;"
                        if val == "evolve":   return "color: #1d4ed8; font-weight: bold;"
                        if val == "merge":    return "color: #92400e; font-weight: bold;"
                        if val == "disappear":return "color: #991b1b; font-weight: bold;"
                        return ""
                    
                    cols_to_show = ["topic_id", "transition", "words", "category", "best_match_topic", "best_match_sim"]
                    avail_cols = [c for c in cols_to_show if c in ex_df.columns]
                    
                    st_df = ex_df[avail_cols].sort_values(["topic_id", "transition"])
                    
                    if "best_match_sim" in st_df.columns:
                        styled_df = st_df.style.map(_style_category, subset=["category"]).format({"best_match_sim": "{:.4f}"}, na_rep="—")
                    else:
                        styled_df = st_df.style.map(_style_category, subset=["category"])
                        
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Topic 0 and 1 have no transition records.")


with tab_compare:
    st.subheader("Average Continuity Rate per Model")
    cont_data = []
    for m in selected_models:
        df = load_csv(m, "consistency", subject, "continuity_summary.csv")
        if df is not None and len(df) > 0:
            row = {"model": MODEL_LABELS[m]}
            if "pct_stable"   in df.columns: row["Stable %"]    = df["pct_stable"].mean()
            if "pct_evolve"   in df.columns: row["Evolve %"]    = df["pct_evolve"].mean()
            if "pct_merge"    in df.columns: row["Merge %"]     = df["pct_merge"].mean()
            if "pct_disappear"in df.columns: row["Disappear %"] = df["pct_disappear"].mean()
            cont_data.append(row)

    if cont_data:
        cd = pd.DataFrame(cont_data)
        # Ordered category list (only those present)
        _cat_order = [c for c in ["Stable %", "Evolve %", "Merge %", "Disappear %"] if c in cd.columns]
        cd_melt = cd.melt(id_vars="model", value_vars=_cat_order, var_name="Category", value_name="Percentage")
        _cat_colors_cmp = {
            "Stable %":    "#22c55e",
            "Evolve %":    "#3b82f6",
            "Merge %":     "#f59e0b",
            "Disappear %": "#ef4444",
        }
        fig = px.bar(
            cd_melt, x="model", y="Percentage", color="Category",
            barmode="stack",
            color_discrete_map=_cat_colors_cmp,
            title=f"Average Continuity Rate by Model — {SUBJECT_LABELS[subject]}",
            category_orders={"Category": _cat_order},
        )
        fig.update_layout(
            height=420,
            yaxis=dict(range=[0, 100], title="Percentage (%)"),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Grouped bar for side-by-side comparison
        fig_grp = px.bar(
            cd_melt, x="model", y="Percentage", color="Category",
            barmode="group",
            color_discrete_map=_cat_colors_cmp,
            title=f"Continuity Category Breakdown (Grouped) — {SUBJECT_LABELS[subject]}",
            category_orders={"Category": _cat_order},
        )
        fig_grp.update_layout(
            height=380,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_grp, use_container_width=True)

        ov_data = []
        for m in selected_models:
            df = load_csv(m, "consistency", subject, "continuity_overall.csv")
            if df is not None and len(df) > 0:
                row = df.iloc[0].to_dict()
                row["model"] = MODEL_LABELS[m]
                ov_data.append(row)
        if ov_data:
            ov_df = pd.DataFrame(ov_data)
            # Preferred column order
            _pref_cols = ["model", "threshold_low", "threshold_high",
                          "avg_pct_stable", "avg_pct_evolve", "avg_pct_merge",
                          "avg_pct_disappear", "total_merge_groups", "total_new"]
            cols = [c for c in _pref_cols if c in ov_df.columns] + \
                   [c for c in ov_df.columns if c not in _pref_cols and c not in ("subject",)]
            st.dataframe(
                ov_df[cols].set_index("model").style.format(
                    {c: "{:.2f}" for c in cols if c != "model" and ov_df[c].dtype in ("float64", "float32")},
                    na_rep="—"
                ),
                use_container_width=True,
            )

    # ── Summary table: all avg rates + new topics ─────────────────────────
    st.subheader("📊 Average Continuity Rates per Model")
    st.caption(
        "Averages computed across all yearly transitions. "
        "**New Topic %** = n_new / n_topics_t1 × 100 per year, then averaged."
    )
    avg_new_rows = []
    for m in selected_models:
        df = load_csv(m, "consistency", subject, "continuity_summary.csv")
        if df is None or len(df) == 0:
            continue
        row_data = {"Model": MODEL_LABELS[m]}
        if "pct_stable"    in df.columns: row_data["Avg Stable %"]    = df["pct_stable"].mean()
        if "pct_evolve"    in df.columns: row_data["Avg Evolve %"]    = df["pct_evolve"].mean()
        if "pct_merge"     in df.columns: row_data["Avg Merge %"]     = df["pct_merge"].mean()
        if "pct_disappear" in df.columns: row_data["Avg Disappear %"] = df["pct_disappear"].mean()
        if "n_new" in df.columns and "n_topics_t1" in df.columns:
            pct_new_series = df["n_new"] / df["n_topics_t1"] * 100
            row_data["Avg New Topic %"]    = pct_new_series.mean()
            row_data["Median New Topic %"] = pct_new_series.median()
            row_data["Min New Topic %"]    = pct_new_series.min()
            row_data["Max New Topic %"]    = pct_new_series.max()
        elif "n_new" in df.columns:
            row_data["Avg New Topic (count)"]    = df["n_new"].mean()
            row_data["Median New Topic (count)"] = df["n_new"].median()
        avg_new_rows.append(row_data)

    if avg_new_rows:
        avg_new_df = pd.DataFrame(avg_new_rows).set_index("Model")
        float_cols = [c for c in avg_new_df.columns if avg_new_df[c].dtype in ("float64", "float32")]
        fmt = {c: "{:.2f}" for c in float_cols}
        _gradient_cols = [c for c in ["Avg Stable %", "Avg Evolve %", "Avg New Topic %"] if c in avg_new_df.columns]
        st.dataframe(
            avg_new_df.style.format(fmt, na_rep="—").background_gradient(
                subset=_gradient_cols,
                cmap="RdYlGn",
                axis=0,
            ),
            use_container_width=True,
        )

        # ── Stacked bar: Avg New Topic % per model
        if "Avg New Topic %" in avg_new_df.columns:
            new_topic_df = avg_new_df[["Avg New Topic %"]].reset_index()
            new_topic_df.columns = ["model", "Avg New Topic %"]
            fig_new = px.bar(
                new_topic_df.sort_values("Avg New Topic %", ascending=False),
                x="model", y="Avg New Topic %",
                color="model",
                color_discrete_map=colors,
                title=f"Avg New Topic % per Model — {SUBJECT_LABELS[subject]}",
                labels={"model": "Model", "Avg New Topic %": "Avg New Topic (%)"},
                text_auto=".2f",
            )
            fig_new.update_layout(height=360, showlegend=False, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_new, use_container_width=True)
    else:
        st.info("No continuity data found for the selected models.")
