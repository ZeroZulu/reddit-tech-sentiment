"""
Topic Modeling Module
======================

Implements topic discovery using two approaches:
1. LDA (Latent Dirichlet Allocation) via gensim — classic, interpretable
2. BERTopic — transformer-based, often produces more coherent topics

Provides evaluation (coherence scores), visualization, and topic-over-time analysis.

Usage
-----
    from src.topics import TopicModeler

    modeler = TopicModeler()
    df, topics = modeler.fit_lda(df_preprocessed)
    modeler.visualize_topics()
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from gensim import corpora
    from gensim.models import LdaMulticore
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.warning("gensim not installed. Install with: pip install gensim")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logger.warning("BERTopic not installed. Install with: pip install bertopic")

from src.utils import load_config, OUTPUT_DIR


class TopicModeler:
    """
    Topic modeling pipeline supporting LDA and BERTopic.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file.

    Attributes
    ----------
    lda_model : gensim.models.LdaMulticore
        Fitted LDA model.
    bertopic_model : BERTopic
        Fitted BERTopic model.
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary for LDA.
    corpus : list
        Bag-of-words corpus for LDA.
    topic_labels : dict
        Human-readable topic labels.

    Examples
    --------
    >>> modeler = TopicModeler()
    >>> df, topic_info = modeler.fit_lda(df_clean, n_topics=10)
    >>> modeler.print_topics()
    """

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.lda_model = None
        self.bertopic_model = None
        self.dictionary = None
        self.corpus = None
        self.topic_labels = {}
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None

    # ─── LDA Topic Modeling ───────────────────────────────────────────────

    def fit_lda(self, df: pd.DataFrame, text_col: str = "text_lemmatized",
                n_topics: int = 10) -> tuple[pd.DataFrame, list[dict]]:
        """
        Fit an LDA topic model on the text data.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame with lemmatized text.
        text_col : str
            Column name containing lemmatized text.
        n_topics : int
            Number of topics to discover.

        Returns
        -------
        tuple[pd.DataFrame, list[dict]]
            DataFrame with topic assignments, and topic info list.
        """
        if not GENSIM_AVAILABLE:
            logger.error("gensim required for LDA — install it first")
            return df, []

        logger.info(f"Fitting LDA model with {n_topics} topics...")

        # Tokenize for gensim
        texts = df[text_col].apply(lambda x: x.split() if isinstance(x, str) else [])

        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in texts]

        # Train LDA
        lda_config = self.config["topic_modeling"]["lda"]
        self.lda_model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=n_topics,
            passes=lda_config["passes"],
            random_state=lda_config["random_state"],
            workers=2,
        )

        # Assign dominant topic to each document
        topic_assignments = []
        for doc_bow in self.corpus:
            topic_dist = self.lda_model.get_document_topics(doc_bow)
            if topic_dist:
                dominant = max(topic_dist, key=lambda x: x[1])
                topic_assignments.append({
                    "lda_topic": dominant[0],
                    "lda_topic_prob": round(dominant[1], 4),
                })
            else:
                topic_assignments.append({"lda_topic": -1, "lda_topic_prob": 0.0})

        topic_df = pd.DataFrame(topic_assignments)
        df = df.reset_index(drop=True)
        df = pd.concat([df, topic_df], axis=1)

        # Extract topic info
        topic_info = self._extract_lda_topics(n_topics)

        logger.info(f"LDA model fitted — {n_topics} topics discovered")
        return df, topic_info

    def _extract_lda_topics(self, n_topics: int, n_words: int = 10) -> list[dict]:
        """Extract topic keywords from fitted LDA model."""
        topics = []
        for topic_id in range(n_topics):
            words = self.lda_model.show_topic(topic_id, topn=n_words)
            topics.append({
                "topic_id": topic_id,
                "words": [w[0] for w in words],
                "weights": [round(w[1], 4) for w in words],
                "label": f"Topic {topic_id}",
            })
        return topics

    def evaluate_coherence(self, texts: list[list[str]],
                           topic_counts: list[int] = None) -> dict:
        """
        Evaluate LDA coherence scores across different topic counts.

        Parameters
        ----------
        texts : list[list[str]]
            Tokenized documents (list of word lists).
        topic_counts : list[int], optional
            Topic counts to evaluate. Defaults to config values.

        Returns
        -------
        dict
            Mapping of topic count to coherence score.
        """
        if not GENSIM_AVAILABLE:
            return {}

        if topic_counts is None:
            topic_counts = self.config["topic_modeling"]["lda"]["topic_counts"]

        lda_config = self.config["topic_modeling"]["lda"]
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in texts]

        results = {}
        for n in topic_counts:
            logger.info(f"Evaluating coherence for {n} topics...")
            model = LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                num_topics=n,
                passes=lda_config["passes"],
                random_state=lda_config["random_state"],
                workers=2,
            )
            cm = CoherenceModel(
                model=model, texts=texts,
                dictionary=dictionary, coherence="c_v"
            )
            score = cm.get_coherence()
            results[n] = round(score, 4)
            logger.info(f"  n_topics={n} → coherence={score:.4f}")

        return results

    # ─── BERTopic ─────────────────────────────────────────────────────────

    def fit_bertopic(self, df: pd.DataFrame,
                     text_col: str = "text_clean") -> tuple[pd.DataFrame, object]:
        """
        Fit a BERTopic model on the text data.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame.
        text_col : str
            Column name containing text to model.

        Returns
        -------
        tuple[pd.DataFrame, BERTopic]
            DataFrame with BERTopic assignments, and the fitted model.
        """
        if not BERTOPIC_AVAILABLE:
            logger.error("BERTopic required — install with: pip install bertopic")
            return df, None

        bt_config = self.config["topic_modeling"]["bertopic"]
        logger.info("Fitting BERTopic model...")

        self.bertopic_model = BERTopic(
            min_topic_size=bt_config["min_topic_size"],
            nr_topics=bt_config["nr_topics"],
            verbose=True,
        )

        texts = df[text_col].tolist()
        topics, probs = self.bertopic_model.fit_transform(texts)

        df = df.copy()
        df["bertopic_topic"] = topics
        df["bertopic_prob"] = [round(float(p), 4) if p is not None else 0.0
                               for p in probs]

        topic_info = self.bertopic_model.get_topic_info()
        n_topics = len(topic_info[topic_info["Topic"] != -1])
        logger.info(f"BERTopic discovered {n_topics} topics "
                    f"({(np.array(topics) == -1).sum()} outliers)")

        return df, self.bertopic_model

    # ─── TF-IDF (for visualization) ──────────────────────────────────────

    def build_tfidf(self, df: pd.DataFrame,
                    text_col: str = "text_lemmatized",
                    max_features: int = 5000) -> None:
        """
        Build TF-IDF matrix for topic analysis and visualization.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with lemmatized text column.
        text_col : str
            Column with processed text.
        max_features : int
            Maximum vocabulary size.
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for TF-IDF")
            return

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=5,
            max_df=0.5,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            df[text_col].fillna("")
        )
        logger.info(f"TF-IDF matrix built: {self.tfidf_matrix.shape}")

    # ─── Topic Over Time ──────────────────────────────────────────────────

    @staticmethod
    def topics_over_time(df: pd.DataFrame,
                         topic_col: str = "lda_topic",
                         freq: str = "W") -> pd.DataFrame:
        """
        Compute topic prevalence over time periods.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with topic assignments and timestamps.
        topic_col : str
            Column name for topic assignments.
        freq : str
            Resampling frequency ('D', 'W', 'M').

        Returns
        -------
        pd.DataFrame
            Topic counts per time period (long format).
        """
        df_copy = df.copy()
        df_copy["period"] = pd.to_datetime(
            df_copy["created_utc"], utc=True
        ).dt.to_period(freq)

        topic_time = (
            df_copy.groupby(["period", topic_col])
            .size()
            .reset_index(name="count")
        )
        topic_time["period"] = topic_time["period"].astype(str)
        return topic_time

    # ─── Utilities ────────────────────────────────────────────────────────

    def print_topics(self, n_words: int = 8):
        """Print discovered LDA topics with top keywords."""
        if not self.lda_model:
            logger.warning("No LDA model fitted yet")
            return

        print("\n" + "=" * 60)
        print("DISCOVERED TOPICS")
        print("=" * 60)
        for idx, topic in self.lda_model.print_topics(num_words=n_words):
            print(f"\nTopic {idx}: {topic}")
        print("=" * 60)

    def save_model(self, model_type: str = "lda",
                   path: str = None) -> None:
        """Save fitted model to disk."""
        save_dir = Path(path) if path else OUTPUT_DIR / "models"
        save_dir.mkdir(parents=True, exist_ok=True)

        if model_type == "lda" and self.lda_model:
            self.lda_model.save(str(save_dir / "lda_model"))
            if self.dictionary:
                self.dictionary.save(str(save_dir / "lda_dictionary"))
            logger.info(f"LDA model saved to {save_dir}")

        elif model_type == "bertopic" and self.bertopic_model:
            self.bertopic_model.save(str(save_dir / "bertopic_model"))
            logger.info(f"BERTopic model saved to {save_dir}")

    def get_top_words_per_topic(self, n_words: int = 10) -> dict:
        """
        Get top words for each topic as a dictionary.

        Returns
        -------
        dict
            {topic_id: [(word, weight), ...]}
        """
        if not self.lda_model:
            return {}

        result = {}
        for topic_id in range(self.lda_model.num_topics):
            result[topic_id] = self.lda_model.show_topic(topic_id, topn=n_words)
        return result


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import setup_logging, generate_sample_data
    from src.preprocessor import TextPreprocessor
    setup_logging("INFO")

    # Demo pipeline
    df_raw = generate_sample_data(n_posts=500)
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.fit_transform(df_raw)

    modeler = TopicModeler()

    # Simple TF-IDF demo (doesn't require gensim)
    modeler.build_tfidf(df_clean)
    print(f"\nTF-IDF shape: {modeler.tfidf_matrix.shape}")
    print(f"Vocabulary sample: {list(modeler.tfidf_vectorizer.vocabulary_.keys())[:20]}")
