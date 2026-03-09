"""
Shared Keyword SPAN: Cross-model comparison using the SAME keyword set.

Like the paper's Table 6 / Figure 4:
- Select top-N corpus keywords (by frequency)
- Compute SPAN for each keyword against EACH model
- Save side-by-side comparison CSV

Output:
  results/shared/tren/{subject}/shared_keyword_span.csv
  results/shared/tren/{subject}/shared_keyword_span_summary.csv
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/nedo/Kuliah/TA/Program")
DATA_DIR = BASE / "data" / "preprocess"
RESULTS_DIR = BASE / "results"

LIST_SUBJECT = ["cs", "math", "physics"]
MODELS = ["dtm", "lda", "top2vec", "bertopic"]
MODEL_LABELS = {"dtm": "DTM", "lda": "LDA", "top2vec": "Top2Vec", "bertopic": "BERTopic"}

TOP_N = 50  # number of shared keywords to evaluate


def compute_corpus_word_freq(subject):
    """Count each word across all documents in corpus."""
    df = pd.read_csv(DATA_DIR / subject / "bow/v1.csv")
    word_freq = defaultdict(int)
    for text_val in df["text"]:
        try:
            tokens = ast.literal_eval(text_val)
            if isinstance(tokens, list):
                for w in tokens:
                    word_freq[w] += 1
        except (ValueError, SyntaxError):
            for w in str(text_val).split():
                word_freq[w] += 1
    return word_freq


def load_topic_words_by_year(model, subject):
    """Load topic-word evolution and return {year: [set of words, ...]}."""
    evo_path = RESULTS_DIR / model / "temporal" / subject / "topic_word_evolution.csv"
    if not evo_path.exists():
        return {}, []
    evo_df = pd.read_csv(evo_path)
    years = sorted(evo_df["year"].unique())
    topic_words_by_year = defaultdict(list)
    for _, row in evo_df.iterrows():
        words = set(w.strip() for w in str(row["top_words"]).split(","))
        topic_words_by_year[int(row["year"])].append(words)
    return topic_words_by_year, years


def compute_span(keyword, topic_words_by_year, years):
    """SPAN = longest consecutive years keyword appears in any topic."""
    trend = []
    for y in years:
        found = any(keyword in words for words in topic_words_by_year.get(y, []))
        trend.append(1 if found else 0)
    max_span = 0
    current = 0
    for t in trend:
        if t == 1:
            current += 1
            max_span = max(max_span, current)
        else:
            current = 0
    return max_span, trend


for subject in LIST_SUBJECT:
    print(f"\n{'='*70}")
    print(f"Shared Keyword SPAN: {subject.upper()}")
    print(f"{'='*70}")

    # 1. Select shared keywords: top-N by corpus frequency
    corpus_freq = compute_corpus_word_freq(subject)
    top_keywords = sorted(corpus_freq.items(), key=lambda x: -x[1])[:TOP_N]
    shared_keywords = [w for w, _ in top_keywords]
    print(f"  Top {TOP_N} corpus keywords selected")

    # 2. Load all models
    model_data = {}
    for model in MODELS:
        tw, years = load_topic_words_by_year(model, subject)
        model_data[model] = (tw, years)

    # Use DTM years as reference (all should be same)
    ref_years = model_data["dtm"][1]
    print(f"  Years: {ref_years[0]}–{ref_years[-1]} ({len(ref_years)} years)")

    # 3. Compute SPAN for each keyword × each model
    rows = []
    for word in shared_keywords:
        v_hat = corpus_freq[word]
        row = {"word": word, "v_hat": v_hat}

        for model in MODELS:
            tw, years = model_data[model]
            span, trend = compute_span(word, tw, years)
            s_dict = span / v_hat if v_hat > 0 else 0.0
            label = MODEL_LABELS[model]
            row[f"span_{label}"] = span
            row[f"trend_{label}"] = str(trend)
            row[f"s_dict_{label}"] = round(s_dict, 6)

        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Save
    out_dir = RESULTS_DIR / "shared" / "tren" / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / "shared_keyword_span.csv", index=False)

    # 4. Summary: avg-SPAN per model using SAME keyword set
    #    Paper formula: avg-SPAN = (1/N) × Σ (Sₖ / v̂ₖ)
    summary_rows = []
    for model in MODELS:
        label = MODEL_LABELS[model]
        spans = result_df[f"span_{label}"]
        s_dicts = result_df[f"s_dict_{label}"]
        summary_rows.append({
            "model": label,
            "n_keywords": TOP_N,
            "avg_span_paper": round(s_dicts.mean(), 6),   # paper: mean of Sₖ/v̂ₖ
            "avg_span_simple": round(spans.mean(), 4),     # simple mean of SPAN
            "sum_s_dict": round(s_dicts.sum(), 6),
            "max_span": int(spans.max()),
            "n_captured": int((spans > 0).sum()),
            "n_full_span": int((spans == len(ref_years)).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "shared_keyword_span_summary.csv", index=False)

    # Print results
    print(f"\n  {'Model':<10s} {'avg-SPAN(paper)':>16s} {'avg-SPAN(simple)':>17s} {'Captured':>9s} {'Full':>5s}")
    print(f"  {'-'*62}")
    for _, s in summary_df.iterrows():
        print(f"  {s['model']:<10s} {s['avg_span_paper']:>16.6f} {s['avg_span_simple']:>17.4f} "
              f"{s['n_captured']:>5d}/{TOP_N} {s['n_full_span']:>5d}")

    # Top-10 keywords with cross-model comparison
    print(f"\n  Top 10 keywords (by corpus freq):")
    span_cols = [f"span_{MODEL_LABELS[m]}" for m in MODELS]
    print(f"  {'Word':<20s} {'v̂ₖ':>8s}  " + "  ".join(f"{MODEL_LABELS[m]:>8s}" for m in MODELS))
    for _, r in result_df.head(10).iterrows():
        spans_str = "  ".join(f"{r[c]:>8d}" for c in span_cols)
        print(f"  {r['word']:<20s} {r['v_hat']:>8d}  {spans_str}")

    print(f"\n  Saved: {out_dir}")

print("\nDone!")
