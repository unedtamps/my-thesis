# Topic Modeling Thesis Project

A thesis research project exploring various topic modeling techniques for text analysis.

## Project Structure

```
├── dataset/        # Data files
├── eda/            # Exploratory Data Analysis
├── preprocessing/  # Data preprocessing scripts
├── src/            # Source code
│   ├── bertopic/   # BERTopic implementation
│   ├── dtm/        # Dynamic Topic Model
│   ├── lda/        # Latent Dirichlet Allocation
│   └── top2vec/    # Top2Vec implementation
└── archive/        # Archived files
```

## Tech Stack

- Python 3.13+
- BERTopic, LDA, Top2Vec, FASTopic
- Sentence Transformers
- PyTorch
- scikit-learn, Gensim, spaCy

## Setup

This project uses **two separate environments** for different modules:

### Environment 1: uv (for BERTopic, Top2Vec, LDA)
Uses `pyproject.toml` + `uv.lock`

```bash
# Install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv sync
```

### Environment 2: pip (for DTM)
Uses `requirement.txt`

```bash
# Create virtual environment
python -m venv .venv-dtm
source .venv-dtm/bin/activate

# Install dependencies
pip install -r requirement.txt
```