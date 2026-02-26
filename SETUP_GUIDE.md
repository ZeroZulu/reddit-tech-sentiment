# Reddit Tech Sentiment Tracker — Setup & Execution Guide

**Complete step-by-step instructions to get the project running locally on your machine.**

---

## Prerequisites

Before starting, make sure you have:

- **Python 3.9+** installed → Check with: `python --version`
- **pip** package manager → Check with: `pip --version`
- **Git** installed → Check with: `git --version`
- **Jupyter Notebook or JupyterLab** → We'll install this below
- A **code editor** (VS Code recommended)
- ~2GB of free disk space (for ML model downloads)

---

## PHASE 1: Project Setup (10-15 minutes)

### Step 1: Unzip the Project

Unzip `reddit-tech-sentiment.zip` to your preferred location:

```bash
# Example: unzip to your home folder or Desktop
unzip reddit-tech-sentiment.zip -d ~/Projects/
cd ~/Projects/reddit-tech-sentiment
```

### Step 2: Create a Virtual Environment

This keeps the project's packages separate from your system Python:

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# On Mac/Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

> ✅ You'll see `(venv)` at the start of your terminal prompt when activated.
> ⚠️ You need to activate this every time you open a new terminal to work on the project.

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt
```

**If you get errors**, install in stages:

```bash
# Core packages (these almost never fail)
pip install pandas numpy matplotlib plotly streamlit pyyaml python-dotenv loguru tqdm sqlalchemy

# NLP packages
pip install vaderSentiment nltk spacy regex emoji

# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Machine Learning (may take a few minutes)
pip install scikit-learn gensim wordcloud

# Transformers (large download ~500MB — optional but recommended)
pip install transformers torch

# BERTopic (optional — has many sub-dependencies)
pip install bertopic

# Jupyter
pip install notebook jupyterlab

# Visualization extras
pip install seaborn pyLDAvis
```

### Step 4: Verify Installation

Run this quick check:

```bash
python -c "
import pandas, numpy, plotly, streamlit, yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
print('✅ All core packages installed successfully!')
try:
    from transformers import pipeline
    print('✅ Transformers available (GPU/CPU sentiment)')
except: print('⚠️  Transformers not installed — VADER-only mode will be used')
try:
    from bertopic import BERTopic
    print('✅ BERTopic available')
except: print('⚠️  BERTopic not installed — LDA-only mode will be used')
"
```

---

## PHASE 2: Reddit API Setup (5 minutes — Optional)

> **You can skip this phase entirely.** The project includes a synthetic data generator
> that lets everything run without API credentials. But if you want real Reddit data
> (which looks much better to recruiters), follow these steps:

### Step 5: Create Reddit API Credentials

1. Go to: https://www.reddit.com/prefs/apps
2. Click **"create another app..."** (scroll to bottom)
3. Fill in:
   - **Name:** `tech-sentiment-tracker`
   - **Type:** Select **"script"**
   - **Description:** `NLP sentiment analysis project`
   - **Redirect URI:** `http://localhost:8080`
4. Click **"create app"**
5. Note down:
   - **Client ID** — the string under the app name (looks like `abc123def456`)
   - **Client Secret** — labeled "secret"

### Step 6: Configure Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit the .env file with your credentials
# Use any text editor:
nano .env        # Linux/Mac
notepad .env     # Windows
code .env        # VS Code
```

Fill in your credentials:

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=tech-sentiment-tracker/1.0 (by u/YOUR_REDDIT_USERNAME)
```

> ⚠️ **NEVER commit the `.env` file to GitHub.** It's already in `.gitignore`.

---

## PHASE 3: Run the Analysis Pipeline (30-45 minutes)

### Step 7: Launch Jupyter Notebook

```bash
# Make sure you're in the project root directory
cd ~/Projects/reddit-tech-sentiment

# Make sure your virtual environment is activated
# (you should see (venv) in your prompt)

# Launch Jupyter
jupyter notebook
```

This opens Jupyter in your browser. Navigate to the `notebooks/` folder.

### Step 8: Run Notebook 01 — Data Collection

**Open:** `notebooks/01_data_collection.ipynb`

**Run each cell top-to-bottom** (Shift+Enter to run a cell):

1. The first cell imports packages and loads config
2. **If you have Reddit API credentials (Phase 2):**
   - Uncomment the cells under "Option A: Live Reddit API Scraping"
   - This will scrape ~5,000 real posts (takes 5-10 minutes due to rate limiting)
3. **If you skipped Phase 2:**
   - Run the cells under "Option B: Synthetic Data for Development"
   - This generates 5,000 realistic fake posts instantly
4. Run the SQLite storage cell — this saves your data to `data/raw/reddit_posts.db`
5. Run the events timeline cell to verify supplementary data loaded

**Expected output:** "Generated/Collected 5,000 posts" and a preview table

> 💡 **Pro tip:** If using real data and want MORE posts, change `n_posts=5000` to
> `limit=2000` per subreddit in the scraper call. More data = better analysis.

### Step 9: Run Notebook 02 — EDA & Preprocessing

**Open:** `notebooks/02_eda_and_preprocessing.ipynb`

Run all cells sequentially. This notebook:

1. Loads raw data and displays basic statistics
2. Shows post distribution by subreddit (bar chart)
3. Shows score and comment distributions (histograms)
4. Shows weekly post volume over time (line chart)
5. Runs the text preprocessing pipeline (cleaning, lemmatization)
6. Shows word count distributions and posting pattern heatmaps
7. **Saves** cleaned data to `data/processed/posts_cleaned.parquet`

**Expected output:** Several interactive Plotly charts + "Saved 4,900+ preprocessed posts"

> ⚠️ If you see a spaCy model error, run: `python -m spacy download en_core_web_sm`

### Step 10: Run Notebook 03 — Sentiment Analysis

**Open:** `notebooks/03_sentiment_analysis.ipynb`

Run all cells sequentially:

1. Loads the cleaned data from notebook 02
2. Applies VADER sentiment analysis (fast, ~2 seconds)
3. Shows sentiment distribution (pie chart + histogram)
4. **For transformer sentiment (optional but recommended):**
   - Uncomment the transformer cells
   - First run will download the DistilBERT model (~250MB)
   - Takes ~1-2 minutes for 5K posts on CPU
5. Compares VADER vs Transformer accuracy
6. Shows sentiment by subreddit (bar chart)
7. Shows weekly sentiment trends (line chart with confidence band)
8. **Saves** to `data/processed/posts_sentiment.parquet`

**Expected output:** Sentiment distributions, comparison metrics, trend charts

### Step 11: Run Notebook 04 — Topic Modeling

**Open:** `notebooks/04_topic_modeling.ipynb`

Run all cells sequentially:

1. Builds TF-IDF matrix and shows top terms
2. **For LDA topics (if gensim installed):**
   - Uncomment the LDA cells
   - Evaluates coherence for different topic counts (takes 2-5 minutes)
   - Fits the optimal model
3. Shows topic distribution bar chart
4. Shows topic × sentiment cross-analysis
5. Shows topic prevalence over time (area chart)
6. **Saves** to `data/processed/posts_final.parquet`

> 💡 If gensim is not installed, the notebook uses simulated topic assignments
> for demonstration purposes. The charts will still look good.

### Step 12: Run Notebook 05 — Trend Analysis

**Open:** `notebooks/05_trend_analysis.ipynb`

Run all cells:

1. Shows sentiment trend with tech event annotations (key chart!)
2. Shows per-subreddit monthly trends
3. Runs rolling sentiment with anomaly detection
4. Computes engagement-sentiment correlation (scatter + stats)

**Expected output:** The annotated trend chart is the most impressive visualization — make sure it renders properly.

### Step 13: Run Notebook 06 — Insights & Findings

**Open:** `notebooks/06_insights_and_findings.ipynb`

This is a summary notebook with markdown only (no code to run). Review the findings
and **update them with your actual numbers** from the previous notebooks.

---

## PHASE 4: Launch the Dashboard (5 minutes)

### Step 14: Run the Streamlit Dashboard

```bash
# Make sure you're in the project root
cd ~/Projects/reddit-tech-sentiment

# Make sure venv is activated
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Launch the dashboard
streamlit run app/streamlit_app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

**Explore the 5 tabs:**

1. **📈 Overview** — Key metrics, sentiment pie chart, subreddit distribution
2. **🔄 Sentiment Trends** — Time series with daily/weekly/monthly toggle
3. **🏢 Company Tracker** — Select companies, see mention volume and sentiment
4. **📊 Subreddit Analysis** — Box plots, scatter plots, posting heatmap
5. **🔍 Data Explorer** — Searchable table with color-coded sentiment

**Use the sidebar filters** to drill down by subreddit, date range, sentiment, and score.

> 💡 Take screenshots of the dashboard for the README and your portfolio!
> Press `Ctrl+Shift+S` in Chrome for a full-page screenshot.

### Step 15: Stop the Dashboard

Press `Ctrl+C` in the terminal where Streamlit is running.

---

## PHASE 5: Run Tests (2 minutes)

### Step 16: Run Unit Tests

```bash
cd ~/Projects/reddit-tech-sentiment
pytest tests/ -v
```

**Expected output:** All tests should pass with green checkmarks. If any fail,
note which ones — we can fix them together.

For a coverage report:

```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing -v
```

---

## PHASE 6: Push to GitHub (10 minutes)

### Step 17: Create GitHub Repository

1. Go to https://github.com/new
2. **Repository name:** `reddit-tech-sentiment`
3. **Description:** `NLP Sentiment & Topic Analysis — Reddit Tech Industry Tracker`
4. Set to **Public**
5. Do **NOT** initialize with README (we already have one)
6. Click "Create repository"

### Step 18: Push Your Code

```bash
cd ~/Projects/reddit-tech-sentiment

# Initialize git
git init

# Add all files (respects .gitignore)
git add .

# Verify what's being tracked (make sure .env is NOT listed)
git status

# First commit
git commit -m "Initial commit: NLP sentiment & topic analysis pipeline"

# Connect to GitHub (replace with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/reddit-tech-sentiment.git

# Push
git branch -M main
git push -u origin main
```

### Step 19: Update README Links

Edit `README.md` and replace placeholder links with your actual GitHub username:

- `https://github.com/shrilpatel` → your actual GitHub profile
- `https://linkedin.com/in/shrilpatel` → your actual LinkedIn

```bash
git add README.md
git commit -m "Update profile links"
git push
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'xxx'` | Run `pip install xxx` with your venv activated |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` |
| spaCy download fails behind firewall | Run `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz` |
| `PRAW` authentication error | Double-check your `.env` credentials and make sure the Reddit app type is "script" |
| Streamlit won't open browser | Manually go to `http://localhost:8501` |
| `torch` installation too large | Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only (smaller) |
| Notebook can't find `src` module | Make sure you're running Jupyter from the project root directory |
| `FileNotFoundError` for parquet files | Run notebooks in order: 01 → 02 → 03 → 04 → 05 |
| Dashboard shows no data | The dashboard uses synthetic data by default — this is expected |
| BERTopic installation fails | Skip it — the project works without it using LDA or simulated topics |

### Getting Help

If you run into issues, take a screenshot of the error and upload it here.
Include:
- Which step you were on
- The full error message
- Your OS (Windows/Mac/Linux)

---

## What to Upload for Review

Once you've run everything, upload these to our chat for review:

1. **All 6 executed notebooks** (File → Download as → Notebook (.ipynb))
   - These should have visible outputs/charts in the cells
2. **The processed data files** from `data/processed/`:
   - `posts_cleaned.parquet` (or export as CSV)
   - `posts_sentiment.parquet` (or export as CSV)
3. **Dashboard screenshots** (at least one per tab)
4. **Test results** — copy/paste the pytest output
5. **Any errors or warnings** you encountered

---

## Quick Reference Commands

```bash
# Activate virtual environment
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# Run Jupyter
jupyter notebook

# Run Dashboard
streamlit run app/streamlit_app.py

# Run Tests
pytest tests/ -v

# Run Scraper (with API credentials)
python -m src.scraper

# Deactivate virtual environment
deactivate
```
