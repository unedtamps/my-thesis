import streamlit as st
import pandas as pd
from pathlib import Path

MODELS = ["lda", "top2vec", "dtm", "bertopic"]
MODEL_LABELS = {"lda": "LDA", "top2vec": "Top2Vec", "dtm": "DTM", "bertopic": "BERTopic"}
SUBJECTS = ["cs", "math", "physics"]
SUBJECT_LABELS = {"cs": "Computer Science", "math": "Mathematics", "physics": "Physics"}


def get_results_dir() -> Path:
    """Auto-detect the results directory."""
    here = Path(__file__).resolve().parent.parent
    candidates = [here / "results", here.parent / "results"]
    for c in candidates:
        if c.exists():
            return c
    return here / "results"


RESULTS_DIR = get_results_dir()


@st.cache_data(show_spinner=False)
def load_csv(model: str, scenario: str, subject: str, filename: str) -> pd.DataFrame | None:
    """Load a CSV from results/{model}/{scenario}/{subject}/{filename}."""
    path = RESULTS_DIR / model / scenario / subject / filename
    if path.exists() and path.stat().st_size > 10:
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_tuning(model: str, subject: str) -> pd.DataFrame | None:
    """
    Load tuning results for any model, handling different paths and column names.
    Normalizes columns to: coherence_cv, irbo_mean, num_topics, topic_quality.
    """
    # Model-specific paths
    search_paths = {
        "lda": [RESULTS_DIR / "lda" / "tuning" / subject / "tuning_results.csv"],
        "dtm": [RESULTS_DIR / "dtm" / "tuning" / subject / "tuning_results.csv"],
        "top2vec": [RESULTS_DIR / "top2vec" / "tuning" / "tuning_results.csv"],
        "bertopic": [
            RESULTS_DIR / "bertopic" / "tuning" / "hdbscan" / "quality_results.csv",
            RESULTS_DIR / "bertopic" / "tuning" / "kmeans" / "quality_results.csv",
        ],
    }

    paths = search_paths.get(model, [])
    df = None
    for p in paths:
        if p.exists() and p.stat().st_size > 10:
            df = pd.read_csv(p)
            break

    if df is None or len(df) == 0:
        return None

    # Filter by subject if column exists and CSV has multiple subjects
    if "subject" in df.columns and subject in df["subject"].values:
        df = df[df["subject"] == subject].copy()
    elif "subject" in df.columns:
        return None

    # Normalize column names
    rename = {}
    if "coherence" in df.columns and "coherence_cv" not in df.columns:
        rename["coherence"] = "coherence_cv"
    if "irbo_aggregated" in df.columns and "irbo_mean" not in df.columns:
        rename["irbo_aggregated"] = "irbo_mean"
    if "k" in df.columns and "num_topics" not in df.columns:
        rename["k"] = "num_topics"
    if "n_topics" in df.columns and "num_topics" not in df.columns:
        rename["n_topics"] = "num_topics"
    if rename:
        df = df.rename(columns=rename)

    return df


@st.cache_data(show_spinner=False)
def load_all_models(scenario: str, subject: str, filename: str) -> pd.DataFrame:
    """Load the same CSV from all 4 models and concatenate with a 'model' column."""
    frames = []
    for m in MODELS:
        df = load_csv(m, scenario, subject, filename)
        if df is not None and len(df) > 0:
            df = df.copy()
            df["model"] = MODEL_LABELS[m]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def setup_sidebar():
    """Shared sidebar with model and subject selectors."""
    st.sidebar.title("Filters")
    subject = st.sidebar.selectbox(
        "Subject", SUBJECTS, format_func=lambda x: SUBJECT_LABELS[x]
    )
    models = st.sidebar.multiselect(
        "Models",
        MODELS,
        default=MODELS,
        format_func=lambda x: MODEL_LABELS[x],
    )
    return subject, models


def model_color_map():
    """Consistent colors for each model across all charts."""
    return {
        "LDA": "#ef4444",
        "Top2Vec": "#8b5cf6",
        "DTM": "#f59e0b",
        "BERTopic": "#3b82f6",
    }
