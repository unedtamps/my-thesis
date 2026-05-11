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

st.set_page_config(page_title="Scenario 4: Dynamics and Transitions", page_icon="🔗", layout="wide")
st.title("🔗 Scenario 4: Dynamics and Transitions")
st.markdown("""
Evaluates how structurally stable topics remain over time.

**Continuity Rate** — classifies topic transitions year-by-year (RBO similarity):
- *Disappear*: sim ≤ 0.2 — topic lost, no recognisable successor
- *Merge*: sim > 0.2, but >1 topic maps to the same t+1-topic
- *Mismatch*: sim > 0.2, single source, but best-match ID ≠ own ID (topic drifted to a different slot)
- *Stable*: sim > 0.6, best-match ID = own ID (topic persists clearly)
- *Evolve*: 0.2 < sim ≤ 0.6, best-match ID = own ID (topic recognisable but shifting)
- *New*: t+1-topic has no incoming match with sim > 0.2
""")
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()



# ============================================================
# Continuity Rate
# ============================================================
st.header("Continuity Rate")
st.divider()

tab_timeline, tab_compare = st.tabs(["Timeline", "Model Comparison"])

with tab_timeline:
    for m in selected_models:
        df = load_csv(m, "continuity", subject, "continuity_summary.csv")
        if df is None or len(df) == 0:
            continue

        with st.expander(f"📋 {MODEL_LABELS[m]} — Continuity Timeline", expanded=len(selected_models) == 1):
            df["transition"] = df["year_from"].astype(str) + "→" + df["year_to"].astype(str)

            _cat_colors = [
                ("pct_stable",    "#22c55e", "Stable"),
                ("pct_evolve",    "#3b82f6", "Evolve"),
                ("pct_merge",     "#f59e0b", "Merge"),
                ("pct_mismatch",  "#a855f7", "Mismatch"),
                ("pct_disappear", "#ef4444", "Disappear"),
            ]

            # ── Stacked bar ───────────────────────────────────────────────────
            fig = go.Figure()
            for cat, color, label in _cat_colors:
                if cat in df.columns:
                    fig.add_trace(go.Bar(
                        x=df["transition"], y=df[cat],
                        name=label, marker_color=color,
                    ))
            fig.update_layout(
                barmode="stack", height=420,
                title=f"{MODEL_LABELS[m]}: Continuity Rate Over Time — {SUBJECT_LABELS[subject]}",
                yaxis_title="Percentage (%)",
                yaxis=dict(range=[0, 100]),
                legend=dict(orientation="h", y=-0.2),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)


            # ── New-topic % per transition ────────────────────────────────────
            if "n_new" in df.columns and "n_topics_t1" in df.columns:
                df_new = pd.DataFrame({
                    "transition": df["transition"],
                    "New Topic %": df["n_new"] / df["n_topics_t1"] * 100,
                })
                fig_new = px.line(
                    df_new, x="transition", y="New Topic %", markers=True,
                    title=f"{MODEL_LABELS[m]}: New Topic % per Year-Transition",
                )
                fig_new.update_traces(line_color="#06b6d4", marker=dict(size=7))
                fig_new.update_layout(
                    height=320, yaxis_title="New Topic (%)",
                    xaxis_tickangle=-45, yaxis=dict(range=[0, None]),
                )
                st.plotly_chart(fig_new, use_container_width=True)

            # ── Topic Example: Transition detail (Topic 0 & 1) ───────────────
            st.markdown("##### 🔍 Topic Example: Transition details (Topic 0 & 1)")
            trans_df   = load_csv(m, "continuity", subject, "continuity_transitions.csv")
            merges_df  = load_csv(m, "continuity", subject, "continuity_merges.csv")
            if trans_df is not None and not trans_df.empty:
                ex_df = trans_df[trans_df["topic_id"].isin([0, 1])].copy()
                if not ex_df.empty:
                    ex_df["transition"] = (
                        ex_df["year_from"].astype(str) + " → " + ex_df["year_to"].astype(str)
                    )

                    # ── For merge rows: build a map (year_from, year_to, target) → [sources] ──
                    merge_sources_map = {}
                    if merges_df is not None and not merges_df.empty:
                        import ast
                        for _, mr in merges_df.iterrows():
                            try:
                                srcs = ast.literal_eval(str(mr["source_topics"]))
                            except Exception:
                                srcs = []
                            key = (int(mr["year_from"]), int(mr["year_to"]), int(mr["target_topic"]))
                            merge_sources_map[key] = srcs

                    # ── Build best_match_display column ────────────────────────────────────────
                    def _best_match_display(row):
                        if row.get("category") == "merge" and "best_match_topic" in row:
                            key = (int(row["year_from"]), int(row["year_to"]),
                                   int(row["best_match_topic"]))
                            srcs = merge_sources_map.get(key)
                            if srcs:
                                return f"{srcs} → T{int(row['best_match_topic'])}"
                        if "best_match_topic" in row:
                            return f"T{int(row['best_match_topic'])}"
                        return "—"

                    ex_df["best_match_display"] = ex_df.apply(_best_match_display, axis=1)

                    def _style_category(val):
                        if val == "stable":    return "color:#166534; font-weight:bold;"
                        if val == "evolve":    return "color:#1d4ed8; font-weight:bold;"
                        if val == "merge":     return "color:#92400e; font-weight:bold;"
                        if val == "mismatch":  return "color:#6d28d9; font-weight:bold;"
                        if val == "disappear": return "color:#991b1b; font-weight:bold;"
                        return ""

                    cols_to_show = ["topic_id", "transition", "words", "category",
                                    "best_match_display", "best_match_sim"]
                    avail_cols = [c for c in cols_to_show if c in ex_df.columns]
                    st_df = ex_df[avail_cols].rename(
                        columns={"best_match_display": "best_match_topic"}
                    ).sort_values(["topic_id", "transition"])
                    styled_df = st_df.style.map(_style_category, subset=["category"])
                    if "best_match_sim" in st_df.columns:
                        styled_df = styled_df.format({"best_match_sim": "{:.4f}"}, na_rep="—")
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    st.caption("🔀 **Merge** rows show all co-source topics → target, e.g. `[0, 3, 7] → T5`")
                else:
                    st.info("Topic 0 and 1 have no transition records.")


with tab_compare:
    st.subheader("Average Continuity Rate per Model")

    _CAT_COLORS = {
        "Stable %":    "#22c55e",
        "Evolve %":    "#3b82f6",
        "Merge %":     "#f59e0b",
        "Mismatch %":  "#a855f7",
        "Disappear %": "#ef4444",
    }
    _CAT_LABELS = {
        "pct_stable":    "Stable %",
        "pct_evolve":    "Evolve %",
        "pct_merge":     "Merge %",
        "pct_mismatch":  "Mismatch %",
        "pct_disappear": "Disappear %",
    }

    cont_data = []
    for m in selected_models:
        df = load_csv(m, "continuity", subject, "continuity_summary.csv")
        if df is not None and len(df) > 0:
            row = {"model": MODEL_LABELS[m]}
            for col, lbl in _CAT_LABELS.items():
                if col in df.columns:
                    row[lbl] = df[col].mean()
            cont_data.append(row)

    if cont_data:
        cd = pd.DataFrame(cont_data)
        _cat_order = [lbl for lbl in _CAT_COLORS if lbl in cd.columns]
        cd_melt = cd.melt(id_vars="model", value_vars=_cat_order,
                          var_name="Category", value_name="Percentage")

        # Stacked bar
        fig = px.bar(
            cd_melt, x="model", y="Percentage", color="Category",
            barmode="stack",
            color_discrete_map=_CAT_COLORS,
            title=f"Average Continuity Rate by Model — {SUBJECT_LABELS[subject]}",
            category_orders={"Category": _cat_order},
        )
        fig.update_layout(
            height=460,
            yaxis=dict(range=[0, 100], title="Percentage (%)"),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Grouped bar
        fig_grp = px.bar(
            cd_melt, x="model", y="Percentage", color="Category",
            barmode="group",
            color_discrete_map=_CAT_COLORS,
            title=f"Continuity Category Breakdown (Grouped) — {SUBJECT_LABELS[subject]}",
            category_orders={"Category": _cat_order},
        )
        fig_grp.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_grp, use_container_width=True)

        # Radar chart (multi-model only)
        if len(cont_data) > 1:
            st.subheader("Category Radar Comparison")
            radar_cats = [lbl for lbl in _CAT_COLORS if lbl in cd.columns]
            fig_radar = go.Figure()
            for _, row in cd.iterrows():
                vals = [row.get(c, 0) for c in radar_cats] + [row.get(radar_cats[0], 0)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=radar_cats + [radar_cats[0]],
                    fill="toself", name=row["model"], opacity=0.55,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=440,
                legend=dict(orientation="h", y=-0.1),
                title=f"Continuity Category Radar — {SUBJECT_LABELS[subject]}",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Overall summary table
        ov_data = []
        for m in selected_models:
            df = load_csv(m, "continuity", subject, "continuity_overall.csv")
            if df is not None and len(df) > 0:
                row = df.iloc[0].to_dict()
                row["model"] = MODEL_LABELS[m]
                ov_data.append(row)
        if ov_data:
            ov_df = pd.DataFrame(ov_data)
            _pref_cols = [
                "model",
                "threshold_disappear", "threshold_stable",
                "avg_pct_stable", "avg_pct_evolve", "avg_pct_merge",
                "avg_pct_mismatch", "avg_pct_disappear",
                "total_merge_groups", "total_new",
            ]
            cols = (
                [c for c in _pref_cols if c in ov_df.columns]
                + [c for c in ov_df.columns if c not in _pref_cols and c not in ("subject",)]
            )
            st.dataframe(
                ov_df[cols].set_index("model").style.format(
                    {c: "{:.2f}" for c in cols
                     if c != "model" and c in ov_df.columns
                     and ov_df[c].dtype in ("float64", "float32")},
                    na_rep="—",
                ),
                use_container_width=True,
            )

    # Summary table: avg rates + new topics
    st.subheader("📊 Average Continuity Rates per Model")
    st.caption(
        "Averages across all yearly transitions. "
        "**New Topic %** = n_new / n_topics_t1 × 100 per year, then averaged."
    )
    avg_new_rows = []
    for m in selected_models:
        df = load_csv(m, "continuity", subject, "continuity_summary.csv")
        if df is None or len(df) == 0:
            continue
        row_data = {"Model": MODEL_LABELS[m]}
        for col, lbl in _CAT_LABELS.items():
            if col in df.columns:
                row_data[f"Avg {lbl}"] = df[col].mean()
        if "n_new" in df.columns and "n_topics_t1" in df.columns:
            pct_new_series = df["n_new"] / df["n_topics_t1"] * 100
            row_data["Avg New Topic %"]    = pct_new_series.mean()
            row_data["Median New Topic %"] = pct_new_series.median()
        avg_new_rows.append(row_data)

    if avg_new_rows:
        avg_new_df = pd.DataFrame(avg_new_rows).set_index("Model")
        float_cols = [c for c in avg_new_df.columns if avg_new_df[c].dtype in ("float64", "float32")]
        fmt = {c: "{:.2f}" for c in float_cols}
        _gradient_cols = [c for c in ["Avg Stable %", "Avg New Topic %"] if c in avg_new_df.columns]
        st.dataframe(
            avg_new_df.style.format(fmt, na_rep="—").background_gradient(
                subset=_gradient_cols, cmap="RdYlGn", axis=0,
            ),
            use_container_width=True,
        )

        if "Avg New Topic %" in avg_new_df.columns:
            new_topic_df = avg_new_df[["Avg New Topic %"]].reset_index()
            new_topic_df.columns = ["model", "Avg New Topic %"]
            fig_new = px.bar(
                new_topic_df.sort_values("Avg New Topic %", ascending=False),
                x="model", y="Avg New Topic %",
                color="model", color_discrete_map=colors,
                title=f"Avg New Topic % per Model — {SUBJECT_LABELS[subject]}",
                labels={"model": "Model", "Avg New Topic %": "Avg New Topic (%)"},
                text_auto=".2f",
            )
            fig_new.update_layout(height=360, showlegend=False, yaxis=dict(range=[0, None]))
            st.plotly_chart(fig_new, use_container_width=True)
    else:
        st.info("No continuity data found for the selected models.")
