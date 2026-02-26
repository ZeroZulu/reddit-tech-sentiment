"""
Utility functions for the Reddit Tech Sentiment Tracker.

Handles configuration loading, logging setup, database connections,
and common helper functions used across modules.
"""

import os
import yaml
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ─── Configuration ────────────────────────────────────────────────────────────

def load_config(path: str = None) -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    path : str, optional
        Path to config file. Defaults to config/config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else CONFIG_PATH
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def load_env():
    """Load environment variables from .env file."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Environment variables loaded from .env")
    else:
        logger.warning("No .env file found — using system environment variables")


# ─── Database ─────────────────────────────────────────────────────────────────

def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """
    Create or connect to the SQLite database.

    Parameters
    ----------
    db_path : str, optional
        Path to the database file. Defaults to config setting.

    Returns
    -------
    sqlite3.Connection
        Active database connection.
    """
    if db_path is None:
        config = load_config()
        db_path = PROJECT_ROOT / config["database"]["path"]

    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    logger.info(f"Connected to database: {db_path}")
    return conn


def init_database(conn: sqlite3.Connection) -> None:
    """
    Initialize the database schema for storing Reddit posts.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            subreddit TEXT NOT NULL,
            title TEXT,
            selftext TEXT,
            score INTEGER DEFAULT 0,
            num_comments INTEGER DEFAULT 0,
            created_utc TIMESTAMP,
            upvote_ratio REAL,
            author TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_subreddit ON posts(subreddit);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created ON posts(created_utc);
    """)
    conn.commit()
    logger.info("Database schema initialized")


def db_to_dataframe(conn: sqlite3.Connection, query: str = None) -> pd.DataFrame:
    """
    Load posts from database into a pandas DataFrame.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.
    query : str, optional
        Custom SQL query. Defaults to selecting all posts.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the queried posts.
    """
    if query is None:
        query = "SELECT * FROM posts ORDER BY created_utc DESC"
    df = pd.read_sql_query(query, conn, parse_dates=["created_utc"])
    logger.info(f"Loaded {len(df):,} posts from database")
    return df


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Configure loguru logging.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_file : str, optional
        Path to log file. If None, logs only to stderr.
    """
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    if log_file:
        logger.add(log_file, rotation="10 MB", retention="7 days", level=level)
    logger.info(f"Logging configured — level: {level}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def utc_to_datetime(utc_timestamp: float) -> datetime:
    """Convert a UTC Unix timestamp to a timezone-aware datetime object."""
    return datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)


def detect_company_mentions(text: str, companies: list[dict]) -> list[str]:
    """
    Detect company mentions in a text string.

    Parameters
    ----------
    text : str
        Text to search for company mentions.
    companies : list[dict]
        List of company dicts from config, each with 'name' and 'keywords'.

    Returns
    -------
    list[str]
        List of company names mentioned in the text.
    """
    text_lower = text.lower()
    mentioned = []
    for company in companies:
        if any(kw in text_lower for kw in company["keywords"]):
            mentioned.append(company["name"])
    return mentioned


def generate_sample_data(n_posts: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Reddit post data for development and demo purposes.

    This allows the dashboard and analysis pipeline to run without
    requiring actual Reddit API credentials.

    Parameters
    ----------
    n_posts : int
        Number of synthetic posts to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame of synthetic Reddit posts with realistic distributions.
    """
    import numpy as np

    np.random.seed(seed)

    subreddits = ["technology", "datascience", "MachineLearning",
                  "cscareerquestions", "programming"]

    # Realistic title templates
    title_templates = {
        "technology": [
            "Apple announces {product} with {feature}",
            "Google's new {product} raises privacy concerns",
            "Microsoft acquires {company} for ${amount}B",
            "NVIDIA GPU shortage continues as AI demand surges",
            "EU proposes new {regulation} for Big Tech companies",
            "Meta's {product} struggles to gain user traction",
            "Amazon AWS outage affects thousands of businesses",
            "OpenAI releases {product} — here's what's different",
            "Should we be worried about AI taking our jobs?",
            "Tesla's self-driving {feature} under investigation",
        ],
        "datascience": [
            "What's the best way to learn {skill} in 2025?",
            "I just got rejected from {n} data science jobs — advice?",
            "Is a master's degree still worth it for data science?",
            "How I transitioned from {field} to data science",
            "Python vs R debate in 2025 — which to learn first?",
            "My company wants me to build a {model} — where to start?",
            "Salary thread: How much are data scientists earning in {city}?",
            "Tips for acing the data science technical interview",
            "Should I learn {tool} or {tool2} for my career?",
            "Data science is becoming oversaturated — change my mind",
        ],
        "MachineLearning": [
            "[R] New paper: {model} achieves SOTA on {benchmark}",
            "[D] Is {technique} overhyped or genuinely useful?",
            "[P] I built a {project} using transformers — feedback?",
            "[R] {org} releases open-source {model} with {params}B params",
            "Why does my {model} keep overfitting on {dataset}?",
            "[D] The future of AI regulation — thoughts?",
            "[R] Scaling laws suggest {finding} for LLMs",
            "[P] Real-time {task} using edge deployment",
            "[D] Are we in an AI bubble?",
            "[R] Breakthrough in {area}: {result}",
        ],
        "cscareerquestions": [
            "Is it worth switching from {role1} to {role2} in this market?",
            "Got laid off from {company} — what now?",
            "How to negotiate salary when offered ${amount}k?",
            "Should I join a startup or FAANG as a new grad?",
            "Remote work is dying — how are you all coping?",
            "My manager is pushing me out — how to handle this?",
            "TC: ${amount}k at {company} — should I stay or leave?",
            "Entry-level market is brutal — took {n} months to land a job",
            "Career advice: {age} years old, thinking of switching to tech",
            "Is {language} still worth learning in 2025?",
        ],
        "programming": [
            "Why I switched from {lang1} to {lang2} and never looked back",
            "Unpopular opinion: {technology} is overengineered",
            "What's the most useful programming concept you've learned?",
            "I've been coding for {n} years — here's what I wish I knew",
            "Rust vs Go for backend development — which wins?",
            "{Framework} 5.0 released — here's what's new",
            "The state of {language} in 2025",
            "How do you stay motivated on side projects?",
            "Best practices for {concept} in production",
            "TIL about {concept} — mind blown",
        ],
    }

    fill_values = {
        "product": ["Vision Pro 2", "Pixel 10", "Surface Pro", "Quest 4", "Echo Show"],
        "feature": ["AI integration", "neural engine", "on-device ML", "quantum chip"],
        "company": ["Figma", "Notion", "Discord", "Databricks", "Anthropic"],
        "amount": ["2.5", "6.8", "12", "19", "45"],
        "regulation": ["AI Act", "data privacy law", "antitrust regulation"],
        "skill": ["SQL", "deep learning", "MLOps", "Spark", "LLMs"],
        "field": ["finance", "biology", "marketing", "academia", "consulting"],
        "model": ["transformer", "XGBoost", "random forest", "neural network", "GAN"],
        "tool": ["dbt", "Airflow", "Snowflake", "Databricks", "Kubernetes"],
        "tool2": ["Spark", "Kafka", "Docker", "Terraform", "Tableau"],
        "city": ["NYC", "SF", "Austin", "London", "Toronto", "Seattle"],
        "n": ["50", "100", "200", "6", "18"],
        "org": ["Meta AI", "Google DeepMind", "Anthropic", "Mistral", "xAI"],
        "params": ["7", "13", "70", "180", "405"],
        "benchmark": ["MMLU", "HumanEval", "GSM8K", "HellaSwag", "ARC"],
        "technique": ["RLHF", "LoRA", "RAG", "chain-of-thought", "distillation"],
        "project": ["real-time translator", "code reviewer", "document QA system"],
        "dataset": ["ImageNet", "CIFAR-10", "custom dataset", "tabular data"],
        "finding": ["diminishing returns", "emergent capabilities", "data efficiency"],
        "task": ["object detection", "speech recognition", "anomaly detection"],
        "area": ["protein folding", "robotics", "computer vision", "NLP"],
        "result": ["10x efficiency gain", "superhuman performance", "zero-shot transfer"],
        "role1": ["SWE", "data analyst", "PM", "DevOps", "QA"],
        "role2": ["ML engineer", "data scientist", "SRE", "solutions architect"],
        "language": ["Python", "Rust", "Go", "TypeScript", "Kotlin"],
        "lang1": ["Java", "C++", "PHP", "Ruby", "JavaScript"],
        "lang2": ["Rust", "Go", "Python", "TypeScript", "Kotlin"],
        "Framework": ["React", "Django", "FastAPI", "Next.js", "Vue"],
        "concept": ["dependency injection", "event sourcing", "CQRS", "monads"],
        "technology": ["microservices", "blockchain", "NoSQL", "GraphQL", "Kubernetes"],
        "age": ["28", "32", "35", "40", "45"],
    }

    # Generate dates spanning ~9 months
    start_ts = pd.Timestamp("2024-06-01", tz="UTC")
    end_ts = pd.Timestamp("2025-02-28", tz="UTC")
    date_range = (end_ts - start_ts).total_seconds()

    records = []
    for i in range(n_posts):
        sub = np.random.choice(subreddits, p=[0.25, 0.20, 0.15, 0.25, 0.15])
        template = np.random.choice(title_templates[sub])

        # Fill in template placeholders
        title = template
        for key, values in fill_values.items():
            placeholder = "{" + key + "}"
            if placeholder in title:
                title = title.replace(placeholder, np.random.choice(values), 1)

        # Realistic score distributions (log-normal)
        score = max(0, int(np.random.lognormal(mean=3.5, sigma=1.8)))
        num_comments = max(0, int(score * np.random.uniform(0.1, 2.5)))
        upvote_ratio = round(np.random.beta(8, 2), 2)  # Skewed toward high

        # Timestamp
        random_offset = np.random.uniform(0, date_range)
        created_utc = start_ts + pd.Timedelta(seconds=random_offset)

        records.append({
            "id": f"synth_{i:06d}",
            "subreddit": sub,
            "title": title,
            "selftext": "",  # Titles are primary for demo
            "score": score,
            "num_comments": num_comments,
            "created_utc": created_utc,
            "upvote_ratio": upvote_ratio,
            "author": f"user_{np.random.randint(1, 2000):04d}",
        })

    df = pd.DataFrame(records)
    df = df.sort_values("created_utc").reset_index(drop=True)
    logger.info(f"Generated {len(df):,} synthetic Reddit posts")
    return df
