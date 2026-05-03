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
    "Topic Prevalence": {
        "definition": "Proportion of documents assigned to a topic per year. Shows how popular/dominant a topic is over time.",
        "formula": "Prevalence(t, y) = docs_in_topic_t_year_y / total_docs_year_y",
        "range": "0 to 1",
        "page": "Scenario 2: Evolution",
        "page_link": "/2_Evolution",
    },
    "c-TF-IDF": {
        "definition": "Class-based TF-IDF. Extracts representative words per topic per year by treating all documents in a topic-year as one document and computing TF-IDF.",
        "formula": "c-TF-IDF(w, c) = tf(w, c) × log(1 + A / tf(w, total))",
        "range": "≥ 0",
        "page": "Scenario 2: Evolution",
        "page_link": "/2_Evolution",
    },
    "Slope (linregress)": {
        "definition": "Linear regression slope of topic prevalence over years. Positive = growing topic, negative = declining. Uses all years, not just endpoints.",
        "formula": "prevalence = intercept + slope × year (via scipy.stats.linregress)",
        "range": "Any real number",
        "page": "Scenario 2: Evolution",
        "page_link": "/2_Evolution",
    },
    "TTC (Temporal Topic Coherence)": {
        "definition": "Temporal Topic Coherence: measures how semantically related topic words at time t are with topic words at t+1, evaluated against the full corpus using normalized NPMI. Specifically, all cross-time word pairs (w_i from t) × (w_j from t+1) are scored using NPMI, then normalized to [0, 1] via (NPMI + 1) / 2. High TTC means the topic transitions smoothly and remains meaningful.",
        "formula": "TTC = mean( (NPMI(w_i, w_j) + 1) / 2 )  for w_i ∈ words_t, w_j ∈ words_{t+1}",
        "range": "0 to 1 (higher = better)",
        "page": "Scenario 2: Evolution",
        "page_link": "/2_Evolution",
    },
    "TTS (Temporal Topic Stability)": {
        "definition": "Temporal Topic Stability: measures how similar the ranked word lists of a topic are between consecutive years using Rank-Biased Overlap (RBO). Unlike cosine similarity, RBO is rank-sensitive — words ranked higher contribute more to the score. High TTS means the topic's vocabulary and word ordering remain consistent over time.",
        "formula": "TTS = RBO(ranked_words_t, ranked_words_{t+1}, p=0.9)",
        "range": "0 to 1 (higher = more stable)",
        "page": "Scenario 2: Evolution",
        "page_link": "/2_Evolution",
    },
    "TTQ (Temporal Topic Quality)": {
        "definition": "Temporal Topic Quality: combined evolution quality metric computed as the harmonic mean of TTC and TTS. Using the harmonic mean ensures both coherence and stability must be high simultaneously — a high score in only one dimension will not yield a high TTQ.",
        "formula": "TTQ = 2 × (TTC × TTS) / (TTC + TTS)",
        "range": "0 to 1 (higher = better)",
        "page": "Scenario 2: Evolution",
        "page_link": "/2_Evolution",
    },

    # Scenario 3
    "TTD (Topic Term Drift)": {
        "definition": "Topic Term Drift: measures how much a topic's top-word set has changed between two time points. Each topic's words are encoded as a binary presence vector over the full vocabulary, then the cosine distance between the two vectors is computed. TTD = 0 means the topic vocabulary is identical; TTD = 1 means completely different sets of words.",
        "formula": "TTD = 1 - cosine_sim(binary_word_vector_t1, binary_word_vector_t2)",
        "range": "0 to 1 (0 = no drift, 1 = complete vocabulary replacement)",
        "page": "Scenario 3: Consistency",
        "page_link": "/3_Consistency",
    },
    "YoY Drift": {
        "definition": "Year-over-Year drift: TTD computed between consecutive years only (e.g., 2015→2016). Shows short-term vocabulary changes.",
        "formula": "YoY_drift(t) = TTD(words_t, words_t+1)",
        "range": "0 to 1",
        "page": "Scenario 3: Consistency",
        "page_link": "/3_Consistency",
    },
    "Trajectory Drift": {
        "definition": "TTD computed from the baseline (first year) to each subsequent year. Shows cumulative vocabulary shift over time.",
        "formula": "Trajectory(t) = TTD(words_first_year, words_t)",
        "range": "0 to 1",
        "page": "Scenario 3: Consistency",
        "page_link": "/3_Consistency",
    },
    "Continuity Rate": {
        "definition": "Classifies how topics transition between consecutive years using best-match cosine similarity.",
        "formula": "Each topic at t → best match at t+1. Categories: Stable (1:1), Merge (many:1), Disappear (no match), New (unmatched at t+1)",
        "range": "Categorical",
        "page": "Scenario 3: Consistency",
        "page_link": "/3_Consistency",
    },

    # Scenario 4
    "avg-SPAN (Paper Formula)": {
        "definition": "Averaged SPAN over all unique topic-terms discovered by the model. Each keyword's SPAN is normalized by its total corpus frequency (v̂ₖ) to reward models that capture rare but persistently growing terms.",
        "formula": "avg-SPAN = (1/||Q̂||) × Σ (Sₖ / v̂ₖ)",
        "range": "≥ 0 (higher = better)",
        "page": "Scenario 4: Keyword Trends",
        "page_link": "/4_Keyword_Trends",
    },
    "Shared Keyword SPAN": {
        "definition": "Apple-to-apple cross-model comparison evaluating the same pool of intersection keywords. Keywords are selected if they are captured by at least 2 different models.",
        "formula": "avg-SPAN over the shared keyword pool",
        "range": "≥ 0 (higher = better)",
        "page": "Scenario 4: Keyword Trends",
        "page_link": "/4_Keyword_Trends",
    },
    "Trend Category SPAN": {
        "definition": "Model evaluation against exactly 90 ground-truth keywords categorized by TF-IDF linear regression slope.",
        "formula": "avg-SPAN evaluated precisely on 30 Emerging, 30 Stable, and 30 Decaying keywords",
        "range": "≥ 0 (higher = better)",
        "page": "Scenario 4: Keyword Trends",
        "page_link": "/4_Keyword_Trends",
    },
    "SPAN": {
        "definition": "Longest consecutive sequence of years a keyword appears in any topic's top words. Measures how well a model retains important keywords.",
        "formula": "SPAN(k) = max length of consecutive 1s in presence vector of keyword k",
        "range": "0 to total_years (higher = better retention)",
        "page": "Scenario 4: Keyword Trends",
        "page_link": "/4_Keyword_Trends",
    },
    "Emerging Keywords": {
        "definition": "Words with strong positive TF-IDF slope — low importance early (2000–2010) but growing significance in later years (2015–2025).",
        "formula": "Classified by highest positive linregress slope of TF-IDF over years",
        "range": "Top 30 by slope",
        "page": "Scenario 4: Keyword Trends",
        "page_link": "/4_Keyword_Trends",
    },
    "Stable Keywords": {
        "definition": "Words with near-zero slope and high average TF-IDF — consistently important across all years.",
        "formula": "Ranked by: avg_tfidf / (1 + |slope| × 10000)",
        "range": "Top 30 by stability score",
        "page": "Scenario 4: Keyword Trends",
        "page_link": "/4_Keyword_Trends",
    },
    "Decaying Keywords": {
        "definition": "Words with strong negative TF-IDF slope — high importance early but declining significance in later years.",
        "formula": "Classified by most negative linregress slope",
        "range": "Top 30 by slope",
        "page": "Scenario 4: Keyword Trends",
        "page_link": "/4_Keyword_Trends",
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
