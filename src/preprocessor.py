"""
Text Preprocessing Module
==========================

Handles all text cleaning, normalization, tokenization, and feature
engineering for Reddit post data before NLP analysis.

Pipeline Steps
--------------
1. Combine title + selftext into a single text field
2. Remove URLs, code blocks, special characters
3. Normalize Unicode, lowercase
4. Lemmatize using spaCy
5. Remove stopwords (NLTK + custom Reddit-specific)
6. Engineer numeric features (length, time, engagement)

Usage
-----
    from src.preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    df_clean = preprocessor.fit_transform(df_raw)
"""

import re
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from nltk.corpus import stopwords as nltk_stopwords
    import nltk
    nltk.download("stopwords", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from src.utils import load_config


class TextPreprocessor:
    """
    Text preprocessing pipeline for Reddit posts.

    Cleans raw text, extracts features, and prepares data for
    sentiment analysis and topic modeling.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file.

    Examples
    --------
    >>> preprocessor = TextPreprocessor()
    >>> df_clean = preprocessor.fit_transform(df_raw)
    >>> print(df_clean[["text_clean", "word_count", "sentiment_ready"]].head())
    """

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.nlp = None
        self.stop_words = set()

        self._setup_spacy()
        self._setup_stopwords()

    def _setup_spacy(self):
        """Load the spaCy language model."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not installed — using basic tokenization. "
                           "Install with: pip install spacy && "
                           "python -m spacy download en_core_web_sm")
            return

        model_name = self.config["preprocessing"]["spacy_model"]
        try:
            self.nlp = spacy.load(model_name, disable=["ner", "parser"])
            logger.info(f"spaCy model loaded: {model_name}")
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. "
                           f"Run: python -m spacy download {model_name}")

    def _setup_stopwords(self):
        """Build combined stopword set from NLTK + custom config."""
        if NLTK_AVAILABLE:
            self.stop_words = set(nltk_stopwords.words("english"))
        else:
            # Minimal fallback stopword list
            self.stop_words = {
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "can", "shall",
                "i", "me", "my", "we", "our", "you", "your", "he", "she",
                "it", "they", "them", "this", "that", "these", "those",
                "and", "but", "or", "not", "no", "so", "if", "then",
                "than", "too", "very", "just", "about", "above", "after",
                "before", "between", "into", "through", "during", "with",
                "from", "to", "of", "in", "on", "at", "for", "by", "up",
            }

        # Add custom Reddit-specific stopwords
        custom = self.config["preprocessing"].get("custom_stopwords", [])
        self.stop_words.update(custom)
        logger.info(f"Stopword list built: {len(self.stop_words)} words")

    # ─── Text Cleaning ────────────────────────────────────────────────────

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r"http\S+|www\.\S+", "", text)

    @staticmethod
    def remove_code_blocks(text: str) -> str:
        """Remove markdown code blocks and inline code."""
        # Multi-line code blocks
        text = re.sub(r"```[\s\S]*?```", " ", text)
        # Inline code
        text = re.sub(r"`[^`]+`", " ", text)
        # Indented code blocks (4+ spaces at line start)
        text = re.sub(r"(?m)^ {4,}.*$", " ", text)
        return text

    @staticmethod
    def remove_special_chars(text: str) -> str:
        """Remove special characters, keeping alphanumeric and basic punctuation."""
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r"[^\w\s.,!?;:'-]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def remove_reddit_artifacts(text: str) -> str:
        """Remove Reddit-specific formatting and artifacts."""
        # Remove markdown links [text](url)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove subreddit references
        text = re.sub(r"r/\w+", "", text)
        # Remove user references
        text = re.sub(r"u/\w+", "", text)
        # Remove common Reddit prefixes
        text = re.sub(r"^\s*\[(R|D|P|N)\]\s*", "", text)
        # Remove "Edit:" prefixes
        text = re.sub(r"(?i)\bedit\s*\d*\s*:", "", text)
        return text

    def clean_text(self, text: str) -> str:
        """
        Apply the full text cleaning pipeline.

        Parameters
        ----------
        text : str
            Raw text from a Reddit post.

        Returns
        -------
        str
            Cleaned text ready for NLP processing.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.remove_urls(text)
        text = self.remove_code_blocks(text)
        text = self.remove_reddit_artifacts(text)
        text = self.remove_special_chars(text)
        text = text.lower().strip()

        return text

    def lemmatize(self, text: str) -> str:
        """
        Lemmatize text using spaCy, removing stopwords.

        Parameters
        ----------
        text : str
            Cleaned text to lemmatize.

        Returns
        -------
        str
            Lemmatized text with stopwords removed.
        """
        if not text:
            return ""

        if self.nlp:
            doc = self.nlp(text)
            tokens = [
                token.lemma_
                for token in doc
                if (token.lemma_ not in self.stop_words
                    and not token.is_punct
                    and not token.is_space
                    and len(token.lemma_) > 2)
            ]
        else:
            # Fallback: simple whitespace tokenization + stopword removal
            tokens = [
                w for w in text.split()
                if w not in self.stop_words and len(w) > 2
            ]

        return " ".join(tokens)

    # ─── Feature Engineering ──────────────────────────────────────────────

    @staticmethod
    def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer numeric features from text and metadata.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'text_clean', 'score', 'num_comments', 'created_utc'.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional feature columns.
        """
        # Text length features
        df["char_count"] = df["text_clean"].str.len()
        df["word_count"] = df["text_clean"].str.split().str.len()
        df["avg_word_length"] = df["text_clean"].apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.strip() else 0
        )

        # Time features from created_utc
        if "created_utc" in df.columns:
            ts = pd.to_datetime(df["created_utc"], utc=True)
            df["hour_of_day"] = ts.dt.hour
            df["day_of_week"] = ts.dt.dayofweek  # 0=Monday
            df["month"] = ts.dt.month
            df["year_month"] = ts.dt.to_period("M").astype(str)
            df["week"] = ts.dt.isocalendar().week.astype(int)

        # Engagement features
        if "score" in df.columns and "num_comments" in df.columns:
            df["engagement_ratio"] = (
                df["num_comments"] / df["score"].clip(lower=1)
            ).round(3)
            df["log_score"] = np.log1p(df["score"])
            df["log_comments"] = np.log1p(df["num_comments"])

        return df

    # ─── Main Pipeline ────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame with Reddit post data.

        Returns
        -------
        pd.DataFrame
            Cleaned, tokenized, and feature-engineered DataFrame.
        """
        logger.info(f"Starting preprocessing pipeline — {len(df):,} posts")
        df = df.copy()

        # Combine title and selftext
        df["text_raw"] = (
            df["title"].fillna("") + " " + df["selftext"].fillna("")
        ).str.strip()

        # Clean text
        logger.info("Cleaning text...")
        df["text_clean"] = df["text_raw"].apply(self.clean_text)

        # Lemmatize
        logger.info("Lemmatizing...")
        df["text_lemmatized"] = df["text_clean"].apply(self.lemmatize)

        # Filter out very short posts
        min_len = self.config["preprocessing"]["min_text_length"]
        before_filter = len(df)
        df = df[df["text_clean"].str.len() >= min_len].reset_index(drop=True)
        logger.info(f"Filtered {before_filter - len(df)} posts below "
                    f"{min_len} chars — {len(df):,} remaining")

        # Feature engineering
        logger.info("Engineering features...")
        df = self.extract_text_features(df)

        logger.info(f"Preprocessing complete — {len(df):,} posts ready")
        return df


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import setup_logging, generate_sample_data
    setup_logging("INFO")

    # Demo with synthetic data
    df_raw = generate_sample_data(n_posts=100)
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.fit_transform(df_raw)

    print("\nSample preprocessed data:")
    print(df_clean[["subreddit", "text_clean", "word_count"]].head(10))
    print(f"\nTotal posts processed: {len(df_clean):,}")
