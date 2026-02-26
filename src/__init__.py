"""
Reddit Tech Sentiment Tracker
==============================
NLP pipeline for analyzing sentiment and topics in tech-related Reddit posts.

Modules:
    scraper       — Reddit data collection via PRAW
    preprocessor  — Text cleaning and feature engineering
    sentiment     — VADER + Transformer sentiment scoring
    topics        — LDA and BERTopic topic modeling
    utils         — Configuration loading, logging, helpers
"""

__version__ = "1.0.0"
__author__ = "Shril Patel"
