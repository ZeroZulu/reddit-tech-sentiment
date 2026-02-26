"""
Reddit Data Collection Module
==============================

Collects posts from tech-related subreddits using the PRAW library.
Supports incremental scraping, rate limiting, and database storage.

Usage
-----
    from src.scraper import RedditScraper

    scraper = RedditScraper()
    scraper.scrape_subreddits(limit=1000)
    scraper.save_to_database()
"""

import os
import time
import sqlite3
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("PRAW not installed — Reddit API scraping unavailable. "
                    "Install with: pip install praw")

from src.utils import (
    load_config, load_env, get_db_connection,
    init_database, utc_to_datetime
)


class RedditScraper:
    """
    Reddit post scraper for tech-related subreddits.

    Connects to the Reddit API via PRAW, collects posts with metadata,
    and stores them in a SQLite database for downstream analysis.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file.

    Attributes
    ----------
    config : dict
        Loaded configuration settings.
    reddit : praw.Reddit
        Authenticated Reddit API instance.
    posts : list[dict]
        Collected post data before database insertion.

    Examples
    --------
    >>> scraper = RedditScraper()
    >>> scraper.scrape_subreddits(limit=500)
    >>> scraper.save_to_database()
    >>> df = scraper.to_dataframe()
    """

    def __init__(self, config_path: str = None):
        load_env()
        self.config = load_config(config_path)
        self.posts = []
        self.reddit = None

        if PRAW_AVAILABLE:
            self._authenticate()

    def _authenticate(self):
        """Authenticate with the Reddit API using credentials from .env."""
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT",
                               "tech-sentiment-tracker/1.0")

        if not client_id or not client_secret:
            logger.error(
                "Reddit API credentials not found. "
                "Create a .env file with REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET. "
                "Get credentials at: https://www.reddit.com/prefs/apps"
            )
            return

        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            # Test authentication
            self.reddit.user.me()
            logger.info("Reddit API authentication successful")
        except Exception as e:
            logger.warning(f"Reddit API auth failed (read-only mode): {e}")
            # Fall back to read-only mode
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )

    def scrape_subreddit(self, subreddit_name: str, limit: int = 1000,
                         sort: str = "top", time_filter: str = "year") -> list[dict]:
        """
        Scrape posts from a single subreddit.

        Parameters
        ----------
        subreddit_name : str
            Name of the subreddit (without r/ prefix).
        limit : int
            Maximum number of posts to collect.
        sort : str
            Sort method: 'hot', 'new', 'top', 'rising'.
        time_filter : str
            Time filter for 'top' sort: 'hour', 'day', 'week', 'month', 'year', 'all'.

        Returns
        -------
        list[dict]
            List of post dictionaries with metadata.
        """
        if not self.reddit:
            logger.error("Reddit API not authenticated — cannot scrape")
            return []

        subreddit = self.reddit.subreddit(subreddit_name)
        scraped_posts = []

        logger.info(f"Scraping r/{subreddit_name} | sort={sort} | limit={limit}")

        try:
            if sort == "top":
                submissions = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort == "hot":
                submissions = subreddit.hot(limit=limit)
            elif sort == "new":
                submissions = subreddit.new(limit=limit)
            elif sort == "rising":
                submissions = subreddit.rising(limit=limit)
            else:
                submissions = subreddit.hot(limit=limit)

            for submission in submissions:
                post = {
                    "id": submission.id,
                    "subreddit": subreddit_name,
                    "title": submission.title,
                    "selftext": submission.selftext or "",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": utc_to_datetime(submission.created_utc),
                    "upvote_ratio": submission.upvote_ratio,
                    "author": str(submission.author) if submission.author else "[deleted]",
                }
                scraped_posts.append(post)

            logger.info(f"Scraped {len(scraped_posts)} posts from r/{subreddit_name}")

        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")

        return scraped_posts

    def scrape_subreddits(self, limit: int = None, sort: str = "top",
                          time_filter: str = "year") -> None:
        """
        Scrape posts from all configured subreddits.

        Parameters
        ----------
        limit : int, optional
            Posts per subreddit. Defaults to config value.
        sort : str
            Sort method for posts.
        time_filter : str
            Time window for 'top' sorting.
        """
        if limit is None:
            limit = self.config["reddit"]["post_limit"]

        subreddits = self.config["reddit"]["subreddits"]

        for sub_name in subreddits:
            posts = self.scrape_subreddit(sub_name, limit, sort, time_filter)
            self.posts.extend(posts)

            # Rate limiting — respect Reddit's API guidelines
            time.sleep(2)

        logger.info(f"Total posts collected: {len(self.posts):,}")

    def save_to_database(self, db_path: str = None) -> int:
        """
        Save collected posts to the SQLite database.

        Parameters
        ----------
        db_path : str, optional
            Database file path. Defaults to config setting.

        Returns
        -------
        int
            Number of new posts inserted (duplicates skipped).
        """
        if not self.posts:
            logger.warning("No posts to save — run scrape_subreddits() first")
            return 0

        conn = get_db_connection(db_path)
        init_database(conn)

        inserted = 0
        cursor = conn.cursor()

        for post in self.posts:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO posts
                    (id, subreddit, title, selftext, score, num_comments,
                     created_utc, upvote_ratio, author)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post["id"], post["subreddit"], post["title"],
                    post["selftext"], post["score"], post["num_comments"],
                    post["created_utc"], post["upvote_ratio"], post["author"],
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.IntegrityError:
                continue  # Skip duplicates

        conn.commit()
        conn.close()
        logger.info(f"Saved {inserted:,} new posts to database "
                    f"({len(self.posts) - inserted} duplicates skipped)")
        return inserted

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert collected posts to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame of all collected posts.
        """
        if not self.posts:
            logger.warning("No posts collected yet")
            return pd.DataFrame()

        df = pd.DataFrame(self.posts)
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
        return df

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the collected data.

        Returns
        -------
        dict
            Statistics including total posts, subreddit breakdown, date range.
        """
        if not self.posts:
            return {"total_posts": 0}

        df = self.to_dataframe()
        return {
            "total_posts": len(df),
            "subreddit_counts": df["subreddit"].value_counts().to_dict(),
            "date_range": {
                "earliest": df["created_utc"].min().isoformat(),
                "latest": df["created_utc"].max().isoformat(),
            },
            "avg_score": round(df["score"].mean(), 1),
            "avg_comments": round(df["num_comments"].mean(), 1),
        }


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging("INFO")

    scraper = RedditScraper()
    scraper.scrape_subreddits(limit=500, sort="top", time_filter="year")
    scraper.save_to_database()

    stats = scraper.get_collection_stats()
    logger.info(f"Collection complete: {stats}")
