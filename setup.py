from setuptools import setup, find_packages

setup(
    name="reddit-tech-sentiment",
    version="1.0.0",
    description="NLP Sentiment & Topic Analysis — Reddit Tech Industry Tracker",
    author="Shril Patel",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "praw",
        "pandas",
        "numpy",
        "spacy",
        "vaderSentiment",
        "transformers",
        "gensim",
        "bertopic",
        "plotly",
        "streamlit",
        "sqlalchemy",
        "python-dotenv",
        "PyYAML",
        "tqdm",
        "loguru",
    ],
)
