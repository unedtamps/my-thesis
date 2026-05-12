import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Glossary", page_icon="📖", layout="wide")
st.title("📖 Glossary")
st.markdown("Technical terms used across this dashboard. Click the **page link** to jump to the relevant scenario.")

st.divider()

GLOSSARY = {
    # Scenario 1
    "Coherence (C_v)": {
        "definition": "Measures how semantically related the top words in a topic are. Higher C_v means the topic words make more sense together. Computed using sliding window, word co-occurrence, and cosine similarity.",
        "formula": "C_v = aggregation of NPMI-based similarities over word pairs",
        "range": "0 to 1 (higher = better)",
        "page": "Scenario 1: Tuning",
        "page_link": "/1_Tuning",
    },
    "IRBO (Inverted Rank-Biased Overlap)": {
        "definition": "Measures topic diversity — how different topics are from each other. High IRBO means topics have distinct vocabularies with minimal overlap.",
        "formula": "IRBO = 1 - RBO(topic_i, topic_j) averaged over all pairs",
        "range": "0 to 1 (higher = more diverse)",
        "page": "Scenario 1: Tuning",
        "page_link": "/1_Tuning",
    },
    "Topic Quality": {
        "definition": "Combined metric balancing coherence and diversity. Uses harmonic mean to ensure both must be high for the combined score to be high.",
        "formula": "TQ = 2 × (Coherence × IRBO) / (Coherence + IRBO)",
        "range": "0 to 1 (higher = better)",
        "page": "Scenario 1: Tuning",
        "page_link": "/1_Tuning",
    },

    # Scenario 2
    "Timestep TQ": {
        "definition": "Combined metric balancing semantic coherence (C_v) and diversity (IRBO) for a specific year. Uses harmonic mean to ensure both must be high for the combined score to be high.",
        "formula": "TQ_year = 2 × (C_v_year × IRBO_year) / (C_v_year + IRBO_year)",
        "range": "0 to 1 (higher = better)",
        "page": "Scenario 2: Timestep Quality",
        "page_link": "/2_Timestep_Quality",
    },
    "Slope (linregress)": {
        "definition": "Linear regression slope of topic prevalence over years. Positive = growing topic, negative = declining. Uses all years, not just endpoints.",
        "formula": "prevalence = intercept + slope × year (via scipy.stats.linregress)",
        "range": "Any real number",
        "page": "Scenario 2: Timestep Quality",
        "page_link": "/2_Timestep_Quality",
    },

    # Scenario 3
    "TTC (Temporal Topic Coherence)": {
        "definition": "Temporal Topic Coherence: measures how semantically related topic words at time t are with topic words at t+1, evaluated against the full corpus using normalized NPMI. Specifically, all cross-time word pairs (w_i from t) × (w_j from t+1) are scored using NPMI, then normalized to [0, 1] via (NPMI + 1) / 2. High TTC means the topic transitions smoothly and remains meaningful.",
        "formula": "TTC = mean( (NPMI(w_i, w_j) + 1) / 2 )  for w_i ∈ words_t, w_j ∈ words_{t+1}",
        "range": "0 to 1 (higher = better)",
        "page": "Scenario 3: Evolution Quality",
        "page_link": "/3_Evolution_Quality",
    },
    "TTS (Temporal Topic Stability)": {
        "definition": "Temporal Topic Stability: measures how similar the ranked word lists of a topic are between consecutive years using Rank-Biased Overlap (RBO). Unlike cosine similarity, RBO is rank-sensitive — words ranked higher contribute more to the score. High TTS means the topic's vocabulary and word ordering remain consistent over time.",
        "formula": "TTS = RBO(ranked_words_t, ranked_words_{t+1}, p=0.9)",
        "range": "0 to 1 (higher = more stable)",
        "page": "Scenario 3: Evolution Quality",
        "page_link": "/3_Evolution_Quality",
    },
    "TTQ (Temporal Topic Quality)": {
        "definition": "Temporal Topic Quality: combined evolution quality metric computed as the harmonic mean of TTC and TTS. Using the harmonic mean ensures both coherence and stability must be high simultaneously — a high score in only one dimension will not yield a high TTQ.",
        "formula": "TTQ = 2 × (TTC × TTS) / (TTC + TTS)",
        "range": "0 to 1 (higher = better)",
        "page": "Scenario 3: Evolution Quality",
        "page_link": "/3_Evolution_Quality",
    },

    # Scenario 4
    "Continuity Rate": {
        "definition": "Classifies how topics transition between consecutive years using best-match similarity.",
        "formula": "Each topic at t → best match at t+1. Categories: Stable, Merge, Mismatch, Disappear, Evolve, New.",
        "range": "Categorical",
        "page": "Scenario 4: Dynamics and Transitions",
        "page_link": "/4_Dynamics_Transitions",
    },

    # Scenario 5
    "Topic Prevalence": {
        "definition": "Proportion of documents assigned to a topic per year. Shows how popular/dominant a topic is over time.",
        "formula": "Prevalence(t, y) = docs_in_topic_t_year_y / total_docs_year_y",
        "range": "0 to 1",
        "page": "Topic Explorer",
        "page_link": "/5_Topic_Explorer",
    },
    "Continuity Timeline": {
        "definition": "Visualizes the life-cycle of a specific topic over time, showing its transitions (Stable, Merge, Disappear, etc.) and best-match topic IDs year-by-year.",
        "formula": "Displays the continuity categories derived from Scenario 4 for the selected topic.",
        "range": "Categorical (Stable, Merge, Mismatch, Disappear, Evolve, New)",
        "page": "Topic Explorer",
        "page_link": "/5_Topic_Explorer",
    },
    "Trend Keywords": {
        "definition": "Displays the top-20 most frequent words from topics that belong to the GROWING, STABLE, or DECLINING categories.",
        "formula": "Aggregated term frequencies grouped by topic trend category for the selected model and subject.",
        "range": "Word frequencies (higher = more frequent)",
        "page": "Topic Explorer",
        "page_link": "/5_Topic_Explorer",
    },
}

# Group by scenario
scenarios = {}
for term, info in GLOSSARY.items():
    page = info["page"]
    if page not in scenarios:
        scenarios[page] = []
    scenarios[page].append((term, info))

# Render
for scenario, terms in scenarios.items():
    st.subheader(scenario)
    for term, info in terms:
        with st.expander(f"**{term}**"):
            st.markdown(info["definition"])
            st.code(info["formula"], language=None)
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Range: {info['range']}")
            with col2:
                st.page_link(f"pages{info['page_link']}.py", label=f"→ Go to {info['page']}", icon="🔗")
    st.divider()
