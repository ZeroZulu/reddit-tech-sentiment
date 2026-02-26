"""
Sentiment Analysis Module
==========================

Implements dual-approach sentiment analysis using:
1. VADER (rule-based) — fast, no GPU required, good baseline
2. HuggingFace Transformer (DistilBERT) — higher accuracy on nuanced text

Compares both methods and provides evaluation metrics.

Usage
-----
    from src.sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    df = analyzer.analyze(df_preprocessed)
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER not installed. Install with: pip install vaderSentiment")

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed. "
                   "Install with: pip install transformers torch")

from src.utils import load_config


class SentimentAnalyzer:
    """
    Multi-method sentiment analyzer for Reddit posts.

    Applies VADER (rule-based) and optionally a transformer model
    to score sentiment, then classifies posts as positive/negative/neutral.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file.
    use_transformer : bool
        Whether to load the transformer model (requires GPU/CPU time).

    Examples
    --------
    >>> analyzer = SentimentAnalyzer(use_transformer=False)
    >>> df = analyzer.analyze(df_clean)
    >>> print(df[["text_clean", "vader_compound", "vader_label"]].head())
    """

    def __init__(self, config_path: str = None, use_transformer: bool = True):
        self.config = load_config(config_path)
        self.vader = None
        self.transformer = None

        self._setup_vader()
        if use_transformer:
            self._setup_transformer()

    def _setup_vader(self):
        """Initialize the VADER sentiment analyzer."""
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded")
        else:
            logger.warning("VADER unavailable — skipping rule-based sentiment")

    def _setup_transformer(self):
        """Load the HuggingFace transformer sentiment model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library unavailable — "
                           "using VADER only")
            return

        model_name = self.config["sentiment"]["transformer_model"]
        try:
            self.transformer = hf_pipeline(
                "sentiment-analysis",
                model=model_name,
                truncation=True,
                max_length=512,
            )
            logger.info(f"Transformer model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")

    # ─── VADER Sentiment ──────────────────────────────────────────────────

    def vader_score(self, text: str) -> dict:
        """
        Compute VADER sentiment scores for a single text.

        Parameters
        ----------
        text : str
            Input text to analyze.

        Returns
        -------
        dict
            Dictionary with 'neg', 'neu', 'pos', 'compound' scores.
        """
        if not self.vader or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self.vader.polarity_scores(text)

    def vader_label(self, compound: float) -> str:
        """
        Classify compound score into sentiment label.

        Parameters
        ----------
        compound : float
            VADER compound score (-1 to 1).

        Returns
        -------
        str
            'positive', 'negative', or 'neutral'.
        """
        thresholds = self.config["sentiment"]["vader_threshold"]
        if compound >= thresholds["positive"]:
            return "positive"
        elif compound <= thresholds["negative"]:
            return "negative"
        return "neutral"

    def apply_vader(self, df: pd.DataFrame,
                    text_col: str = "text_clean") -> pd.DataFrame:
        """
        Apply VADER sentiment to all posts.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a text column.
        text_col : str
            Column name containing text to analyze.

        Returns
        -------
        pd.DataFrame
            DataFrame with VADER score columns added.
        """
        logger.info(f"Applying VADER sentiment to {len(df):,} posts...")

        scores = df[text_col].apply(self.vader_score)
        df["vader_neg"] = scores.apply(lambda x: x["neg"])
        df["vader_neu"] = scores.apply(lambda x: x["neu"])
        df["vader_pos"] = scores.apply(lambda x: x["pos"])
        df["vader_compound"] = scores.apply(lambda x: x["compound"])
        df["vader_label"] = df["vader_compound"].apply(self.vader_label)

        label_counts = df["vader_label"].value_counts()
        logger.info(f"VADER results — "
                    f"Positive: {label_counts.get('positive', 0):,} | "
                    f"Neutral: {label_counts.get('neutral', 0):,} | "
                    f"Negative: {label_counts.get('negative', 0):,}")

        return df

    # ─── Transformer Sentiment ────────────────────────────────────────────

    def apply_transformer(self, df: pd.DataFrame,
                          text_col: str = "text_clean") -> pd.DataFrame:
        """
        Apply HuggingFace transformer sentiment to all posts.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a text column.
        text_col : str
            Column name containing text to analyze.

        Returns
        -------
        pd.DataFrame
            DataFrame with transformer score columns added.
        """
        if not self.transformer:
            logger.warning("Transformer model not loaded — skipping")
            df["transformer_label"] = "unknown"
            df["transformer_score"] = 0.0
            return df

        batch_size = self.config["sentiment"]["batch_size"]
        texts = df[text_col].tolist()

        # Truncate very long texts to avoid memory issues
        texts = [t[:512] if len(t) > 512 else t for t in texts]

        logger.info(f"Running transformer inference on {len(texts):,} posts "
                    f"(batch_size={batch_size})...")

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Handle empty strings
            batch = [t if t.strip() else "neutral" for t in batch]
            try:
                batch_results = self.transformer(batch)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Transformer error at batch {i}: {e}")
                results.extend([{"label": "NEUTRAL", "score": 0.0}] * len(batch))

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"  Processed {min(i + batch_size, len(texts)):,}/{len(texts):,}")

        df["transformer_label"] = [
            "positive" if r["label"] == "POSITIVE" else "negative"
            for r in results
        ]
        df["transformer_score"] = [
            r["score"] if r["label"] == "POSITIVE" else -r["score"]
            for r in results
        ]

        label_counts = df["transformer_label"].value_counts()
        logger.info(f"Transformer results — "
                    f"Positive: {label_counts.get('positive', 0):,} | "
                    f"Negative: {label_counts.get('negative', 0):,}")

        return df

    # ─── Combined Analysis ────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame,
                text_col: str = "text_clean") -> pd.DataFrame:
        """
        Run the complete sentiment analysis pipeline.

        Applies both VADER and transformer (if available), then
        creates an ensemble score combining both methods.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame with text column.
        text_col : str
            Column name containing cleaned text.

        Returns
        -------
        pd.DataFrame
            DataFrame with all sentiment columns added.
        """
        df = df.copy()

        # VADER
        df = self.apply_vader(df, text_col)

        # Transformer (if available)
        if self.transformer:
            df = self.apply_transformer(df, text_col)

            # Ensemble: weighted average (VADER 0.4, Transformer 0.6)
            df["ensemble_score"] = (
                0.4 * df["vader_compound"] +
                0.6 * df["transformer_score"]
            ).round(4)
        else:
            df["ensemble_score"] = df["vader_compound"]

        # Final label from ensemble
        df["sentiment_label"] = df["ensemble_score"].apply(self.vader_label)

        logger.info("Sentiment analysis pipeline complete")
        return df

    # ─── Evaluation ───────────────────────────────────────────────────────

    @staticmethod
    def evaluate(df: pd.DataFrame, true_col: str,
                 pred_col: str) -> dict:
        """
        Evaluate sentiment predictions against ground truth labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with true and predicted label columns.
        true_col : str
            Column name for true labels.
        pred_col : str
            Column name for predicted labels.

        Returns
        -------
        dict
            Accuracy, precision, recall, F1 for each class.
        """
        from sklearn.metrics import classification_report, accuracy_score

        y_true = df[true_col]
        y_pred = df[pred_col]

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        logger.info(f"Evaluation — Accuracy: {accuracy:.3f}")
        return {
            "accuracy": round(accuracy, 4),
            "classification_report": report,
        }

    # ─── Aggregation Helpers ──────────────────────────────────────────────

    @staticmethod
    def sentiment_over_time(df: pd.DataFrame,
                            freq: str = "W") -> pd.DataFrame:
        """
        Aggregate sentiment scores over time periods.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'created_utc' and sentiment columns.
        freq : str
            Resampling frequency: 'D' (daily), 'W' (weekly), 'M' (monthly).

        Returns
        -------
        pd.DataFrame
            Time-aggregated sentiment statistics.
        """
        df_ts = df.set_index(pd.to_datetime(df["created_utc"], utc=True))
        agg = df_ts.resample(freq).agg(
            post_count=("vader_compound", "count"),
            avg_vader=("vader_compound", "mean"),
            median_vader=("vader_compound", "median"),
            std_vader=("vader_compound", "std"),
            avg_score=("score", "mean"),
            avg_comments=("num_comments", "mean"),
        ).dropna()

        return agg.round(4)

    @staticmethod
    def sentiment_by_subreddit(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sentiment statistics grouped by subreddit.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with sentiment columns.

        Returns
        -------
        pd.DataFrame
            Subreddit-level sentiment summary.
        """
        return df.groupby("subreddit").agg(
            post_count=("vader_compound", "count"),
            avg_sentiment=("vader_compound", "mean"),
            pct_positive=("vader_label",
                          lambda x: (x == "positive").mean()),
            pct_negative=("vader_label",
                          lambda x: (x == "negative").mean()),
            avg_score=("score", "mean"),
        ).round(4)


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import setup_logging, generate_sample_data
    from src.preprocessor import TextPreprocessor
    setup_logging("INFO")

    # Demo pipeline
    df_raw = generate_sample_data(n_posts=200)
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.fit_transform(df_raw)

    analyzer = SentimentAnalyzer(use_transformer=False)
    df_sentiment = analyzer.analyze(df_clean)

    print("\nSentiment distribution:")
    print(df_sentiment["sentiment_label"].value_counts())
    print("\nBy subreddit:")
    print(analyzer.sentiment_by_subreddit(df_sentiment))
