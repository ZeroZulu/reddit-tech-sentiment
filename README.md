# ⚡ SentimentPulse — Reddit Tech Sentiment Tracker

**NLP pipeline analyzing 294K+ Reddit posts across 6 tech subreddits — combining VADER and transformer-based sentiment analysis, LDA topic modeling, time-series trend analysis, and an interactive Streamlit dashboard.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-SentimentPulse-C8FF00?style=for-the-badge)](https://YOUR-STREAMLIT-URL.streamlit.app)

---

<div align="center">

### **[→ LAUNCH LIVE DASHBOARD ←](https://reddit-tech-sentiment-vm3mwdf2qp5bnqnfrdeqch.streamlit.app)**

*294K posts · 6 subreddits · Real NLP analysis*

</div>

---

## 🎯 What This Does

Tech sentiment is noisy. This project cuts through it — scraping, cleaning, scoring, and visualizing public opinion across the subreddits where engineers, data scientists, and researchers actually talk.

**The pipeline:**

```
Reddit API (PRAW)  →  SQLite  →  Text Cleaning  →  Sentiment + Topics  →  Dashboard
                                  (spaCy, regex)   (VADER + DistilBERT    (Streamlit
                                                    + LDA)                 + Plotly)
```

**Key findings from 294K posts across 6 subreddits:**

- **DistilBERT outperformed VADER by ~14% accuracy** on tech-specific text — sarcasm and domain jargon are where rule-based models break down
- **LDA topic modeling surfaced 10 distinct themes** (optimal coherence c_v = 0.48), including LLM career impact, MLOps tooling, and return-to-office debates
- **Sentiment varies significantly by subreddit** — r/artificial and r/MachineLearn trend more positive than r/dataengineering, which skews toward complaint-driven posts
- **Posting volume peaks Tuesday–Thursday 10AM–2PM EST**, confirming the audience is working tech professionals

---

## 📊 Dashboard

Five interactive tabs, all filterable by subreddit, date range, sentiment label, and minimum score:

| Tab | What's Inside |
|-----|---------------|
| **Overview** | KPI cards, sentiment donut, subreddit volume bars, score distribution |
| **Sentiment Trends** | Time series by subreddit (daily/weekly/monthly), stacked volume area, sentiment heatmap |
| **Topics** | LDA-discovered topics, lollipop sentiment chart, top 5 topics over time |
| **Subreddits** | Violin plots, engagement vs sentiment scatter, stats table with gradient, activity heatmap |
| **Data Explorer** | Searchable/sortable table with color-coded sentiment, 400-row preview |

---

## 📂 Project Structure

```
reddit-tech-sentiment/
├── config/
│   └── config.yaml                 # Subreddit list, model params, tracking config
├── data/
│   ├── raw/                        # SQLite database (gitignored)
│   ├── processed/                  # posts_final.parquet (294K rows — Git LFS)
│   └── external/                   # Stock prices, event timeline
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_and_preprocessing.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_topic_modeling.ipynb
│   ├── 05_trend_analysis.ipynb
│   └── 06_insights_and_findings.ipynb
├── src/
│   ├── scraper.py                  # Reddit API data collection
│   ├── preprocessor.py             # Text cleaning & feature engineering
│   ├── sentiment.py                # VADER + DistilBERT sentiment scoring
│   ├── topics.py                   # LDA topic modeling
│   └── utils.py                    # Config, database, helpers
├── app/
│   └── streamlit_app.py            # Interactive dashboard (5 tabs)
├── tests/
│   └── test_preprocessor.py
├── outputs/
│   ├── figures/
│   └── models/
├── requirements.txt
├── .streamlit/config.toml
├── .gitignore
├── .gitattributes                  # Git LFS tracking for .parquet
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Collection | PRAW, Reddit API |
| Storage | SQLite, SQLAlchemy |
| Text Processing | spaCy, NLTK, regex |
| Sentiment | VADER (baseline) + DistilBERT (transformer) |
| Topic Modeling | gensim LDA |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Testing | pytest |

---

## 🚀 Quick Start

### Run the Dashboard (fastest)

```bash
git clone https://github.com/ZeroZulu/reddit-tech-sentiment.git
cd reddit-tech-sentiment
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Run the Full Pipeline (notebooks)

```bash
# 1. Set up Reddit API credentials
cp .env.example .env
# Edit .env with your Reddit API keys → https://www.reddit.com/prefs/apps

# 2. Install full dependencies
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm

# 3. Run notebooks sequentially
jupyter notebook notebooks/
```

---

## 🔬 Methodology

### Sentiment Analysis

| Method | Accuracy* | Notes |
|--------|-----------|-------|
| **VADER** | ~72% | Fast, no GPU, interpretable compound scores |
| **DistilBERT** | ~86% | Context-aware, handles sarcasm and tech jargon |
| **Ensemble (0.4V + 0.6T)** | ~84% | Balances speed and accuracy |

*Evaluated on manually labeled tech Reddit posts*

### Topic Modeling

LDA tested with 5, 10, 15, and 20 topics — **10 topics** yielded optimal coherence (c_v = 0.48). Topics include AI/ML career impact, cloud infrastructure debates, open-source tooling, data pipeline engineering, and hiring market sentiment.

---

## 🔍 Data

- **Source:** Reddit API via PRAW — 6 subreddits (r/MachineLearn, r/datascience, r/computerscience, r/artificial, r/analytics, r/dataengineering)
- **Volume:** 294K+ posts
- **Format:** Parquet (processed), SQLite (raw)
- **Note:** `posts_final.parquet` is tracked via Git LFS due to file size

---

## 📝 Skills Demonstrated

NLP (sentiment + topic modeling) · Data Engineering (API → DB → ETL) · Machine Learning (model comparison, evaluation) · Interactive Visualization (Plotly + Streamlit) · Software Engineering (modular code, testing, config management)

---

## 👤 Author

**Shril Patel** — [GitHub](https://github.com/ZeroZulu) · [LinkedIn](https://linkedin.com/in/shril-patel-020504284)

---

## 📄 License

MIT — see [LICENSE](LICENSE)
