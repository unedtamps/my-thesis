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

**Topic Term Drift (TTD)** — measures vocabulary change:
- *Endpoint*: drift between first and last year
- *Trajectory*: cumulative drift from baseline year
- *YoY*: drift between consecutive years

**Continuity Rate** — classifies topic transitions:
- *Stable*: topic maps 1:1 to next year
- *Merge*: multiple topics → one
- *Disappear*: topic has no match
- *New*: topic appears without predecessor

"""
)
st.page_link("pages/Glossary.py", label="📖 See Glossary for detailed definitions", icon=None)


subject, selected_models = setup_sidebar()
colors = model_color_map()

# ============================================================
# Topic Term Drift
# ============================================================
st.header("Topic Term Drift (TTD)")
st.divider()

tab_yoy, tab_traj, tab_endpoint = st.tabs(["Year-over-Year", "Trajectory", "Endpoint Summary"])

with tab_yoy:
    yoy = load_all_models("consistency", subject, "ttd_yoy_avg.csv")
    if len(yoy) > 0:
        yoy_f = yoy[yoy["model"].isin([MODEL_LABELS[m] for m in selected_models])]
        yoy_f["transition"] = yoy_f["year_from"].astype(str) + "→" + yoy_f["year_to"].astype(str)

        fig = px.line(
            yoy_f, x="year_from", y="avg_drift", color="model",
            color_discrete_map=colors,
            title="Average YoY Topic Drift Over Time",
            labels={"avg_drift": "Avg Drift (1 - cosine sim)", "year_from": "Year"},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No YoY drift data available.")

with tab_traj:
    traj = load_all_models("consistency", subject, "ttd_trajectory_avg.csv")
    if len(traj) > 0:
        traj_f = traj[traj["model"].isin([MODEL_LABELS[m] for m in selected_models])]

        fig = px.line(
            traj_f, x="year", y="avg_drift", color="model",
            color_discrete_map=colors,
            title="Trajectory Drift from Baseline Year",
            labels={"avg_drift": "Drift from Baseline", "year": "Year"},
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trajectory data available.")

with tab_endpoint:
    st.subheader("Endpoint Drift (First → Last Year)")
    endpoint_data = []
    for m in selected_models:
        df = load_csv(m, "consistency", subject, "ttd_summary.csv")
        if df is not None and len(df) > 0:
            row = df.iloc[0].to_dict() if len(df) == 1 else df.mean(numeric_only=True).to_dict()
            row["model"] = MODEL_LABELS[m]
            endpoint_data.append(row)

    if endpoint_data:
        ep_df = pd.DataFrame(endpoint_data)
        cols_show = ["model"] + [c for c in ep_df.columns if c != "model" and c != "subject"]
        available = [c for c in cols_show if c in ep_df.columns]
        st.dataframe(ep_df[available].set_index("model"), use_container_width=True)

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
            for cat, color in [("pct_stable", "#22c55e"), ("pct_merge", "#f59e0b"), ("pct_disappear", "#ef4444")]:
                if cat in df.columns:
                    fig.add_trace(go.Bar(
                        x=df["transition"], y=df[cat],
                        name=cat.replace("pct_", "").title(),
                        marker_color=color,
                    ))

            fig.update_layout(
                barmode="stack", height=350,
                title=f"{MODEL_LABELS[m]}: Continuity Rate Over Time",
                yaxis_title="Percentage (%)",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_compare:
    st.subheader("Average Continuity Rate per Model")
    cont_data = []
    for m in selected_models:
        df = load_csv(m, "consistency", subject, "continuity_summary.csv")
        if df is not None and len(df) > 0:
            cont_data.append({
                "model": MODEL_LABELS[m],
                "Stable %": df["pct_stable"].mean(),
                "Merge %": df["pct_merge"].mean(),
                "Disappear %": df["pct_disappear"].mean(),
            })

    if cont_data:
        cd = pd.DataFrame(cont_data)
        cd_melt = cd.melt(id_vars="model", var_name="Category", value_name="Percentage")
        fig = px.bar(
            cd_melt, x="model", y="Percentage", color="Category",
            barmode="group",
            color_discrete_map={"Stable %": "#22c55e", "Merge %": "#f59e0b", "Disappear %": "#ef4444"},
            title="Average Continuity Rate Comparison",
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # Overall summary
        ov_data = []
        for m in selected_models:
            df = load_csv(m, "consistency", subject, "continuity_overall.csv")
            if df is not None and len(df) > 0:
                row = df.iloc[0].to_dict()
                row["model"] = MODEL_LABELS[m]
                ov_data.append(row)
        if ov_data:
            ov_df = pd.DataFrame(ov_data)
            cols = ["model"] + [c for c in ov_df.columns if c not in ["model", "subject"]]
            available = [c for c in cols if c in ov_df.columns]
            st.dataframe(ov_df[available].set_index("model"), use_container_width=True)
