"""
Unit Tests for the TextPreprocessor Module
===========================================

Tests text cleaning functions, feature engineering, and edge cases.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    """Create a TextPreprocessor instance for testing."""
    return TextPreprocessor()


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame for testing."""
    return pd.DataFrame({
        "id": ["t1", "t2", "t3", "t4", "t5"],
        "subreddit": ["technology", "datascience", "programming",
                       "MachineLearning", "cscareerquestions"],
        "title": [
            "Apple announces new AI features for iPhone 16",
            "How I transitioned from biology to data science in 6 months",
            "Why I switched from Java to Rust and never looked back",
            "[R] New paper achieves SOTA on MMLU benchmark",
            "Got laid off from Google — what now?",
        ],
        "selftext": [
            "Apple revealed exciting new features at WWDC 2024...",
            "I was a biology researcher for 5 years. Here's my journey...",
            "",
            "We propose a novel architecture that combines...",
            "After 3 years at Google, I was part of the latest round of layoffs.",
        ],
        "score": [1500, 890, 2300, 450, 3200],
        "num_comments": [340, 220, 510, 89, 1200],
        "created_utc": pd.date_range("2024-09-01", periods=5, freq="7D", tz="UTC"),
        "upvote_ratio": [0.92, 0.88, 0.95, 0.91, 0.87],
        "author": ["user_1", "user_2", "user_3", "user_4", "user_5"],
    })


# ─── Text Cleaning Tests ─────────────────────────────────────────────────────

class TestTextCleaning:
    """Test individual text cleaning functions."""

    def test_remove_urls(self, preprocessor):
        text = "Check this out https://example.com/page and www.reddit.com/r/test"
        result = preprocessor.remove_urls(text)
        assert "https://" not in result
        assert "www." not in result
        assert "Check this out" in result

    def test_remove_code_blocks(self, preprocessor):
        text = "Here's code: ```python\nprint('hello')\n``` and more text"
        result = preprocessor.remove_code_blocks(text)
        assert "print" not in result
        assert "more text" in result

    def test_remove_inline_code(self, preprocessor):
        text = "Use `numpy.array()` for arrays"
        result = preprocessor.remove_code_blocks(text)
        assert "numpy.array()" not in result
        assert "Use" in result

    def test_remove_special_chars(self, preprocessor):
        text = "Hello @world #python $$$money 🚀🎉"
        result = preprocessor.remove_special_chars(text)
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_remove_reddit_artifacts(self, preprocessor):
        text = "[R] Check r/datascience and u/someone posted [this](http://link.com)"
        result = preprocessor.remove_reddit_artifacts(text)
        assert "r/datascience" not in result
        assert "u/someone" not in result
        assert "[R]" not in result
        assert "this" in result  # Link text preserved

    def test_clean_text_full_pipeline(self, preprocessor):
        text = "Check https://example.com — [R] New paper by u/researcher on r/ML"
        result = preprocessor.clean_text(text)
        assert result == result.lower()  # Should be lowercase
        assert "https://" not in result
        assert "u/researcher" not in result

    def test_clean_text_empty_string(self, preprocessor):
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text("   ") == ""

    def test_clean_text_none(self, preprocessor):
        assert preprocessor.clean_text(None) == ""

    def test_remove_edit_prefix(self, preprocessor):
        text = "Edit: I forgot to mention this important thing"
        result = preprocessor.remove_reddit_artifacts(text)
        assert not result.strip().startswith("Edit:")


# ─── Lemmatization Tests ─────────────────────────────────────────────────────

class TestLemmatization:
    """Test text lemmatization and stopword removal."""

    def test_lemmatize_removes_stopwords(self, preprocessor):
        text = "this is a very simple test of the system"
        result = preprocessor.lemmatize(text)
        assert "this" not in result.split()
        assert "is" not in result.split()
        assert "the" not in result.split()

    def test_lemmatize_removes_short_words(self, preprocessor):
        text = "i am an ai ml ds big good data"
        result = preprocessor.lemmatize(text)
        # Words of length 2 or less should be removed
        for word in result.split():
            assert len(word) > 2

    def test_lemmatize_empty(self, preprocessor):
        assert preprocessor.lemmatize("") == ""


# ─── Feature Engineering Tests ────────────────────────────────────────────────

class TestFeatureEngineering:
    """Test numeric feature extraction."""

    def test_text_features_added(self, preprocessor, sample_df):
        df = preprocessor.fit_transform(sample_df)
        assert "char_count" in df.columns
        assert "word_count" in df.columns
        assert "avg_word_length" in df.columns

    def test_time_features_added(self, preprocessor, sample_df):
        df = preprocessor.fit_transform(sample_df)
        assert "hour_of_day" in df.columns
        assert "day_of_week" in df.columns
        assert "month" in df.columns

    def test_engagement_features_added(self, preprocessor, sample_df):
        df = preprocessor.fit_transform(sample_df)
        assert "engagement_ratio" in df.columns
        assert "log_score" in df.columns

    def test_word_count_positive(self, preprocessor, sample_df):
        df = preprocessor.fit_transform(sample_df)
        assert (df["word_count"] > 0).all()

    def test_engagement_ratio_nonnegative(self, preprocessor, sample_df):
        df = preprocessor.fit_transform(sample_df)
        assert (df["engagement_ratio"] >= 0).all()


# ─── Full Pipeline Tests ─────────────────────────────────────────────────────

class TestFullPipeline:
    """Test the complete preprocessing pipeline."""

    def test_fit_transform_returns_dataframe(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_adds_text_columns(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df)
        assert "text_raw" in result.columns
        assert "text_clean" in result.columns
        assert "text_lemmatized" in result.columns

    def test_fit_transform_filters_short_posts(self, preprocessor):
        df = pd.DataFrame({
            "title": ["ok", "This is a normal length post about technology"],
            "selftext": ["", "And it has some body text too"],
            "score": [1, 100],
            "num_comments": [0, 50],
            "created_utc": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "subreddit": ["test", "test"],
        })
        result = preprocessor.fit_transform(df)
        assert len(result) <= len(df)  # Some may be filtered

    def test_no_duplicate_columns(self, preprocessor, sample_df):
        result = preprocessor.fit_transform(sample_df)
        assert len(result.columns) == len(set(result.columns))


# ─── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test handling of edge cases and unusual input."""

    def test_all_empty_text(self, preprocessor):
        df = pd.DataFrame({
            "title": ["", "", ""],
            "selftext": ["", "", ""],
            "score": [0, 0, 0],
            "num_comments": [0, 0, 0],
            "created_utc": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            "subreddit": ["test"] * 3,
        })
        result = preprocessor.fit_transform(df)
        # Should handle gracefully, may return empty DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_unicode_text(self, preprocessor):
        text = "Machine learning est très cool! 机器学习 🤖"
        result = preprocessor.clean_text(text)
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
