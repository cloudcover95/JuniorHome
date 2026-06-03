# path: src/juniorhome/rss_ingester.py
#!/usr/bin/env python3
"""
RSS Ingester with Tagging

Ingests RSS feeds into the Second Brain with automatic tagging.
Supports sovereign, rate-limited ingestion of external knowledge.
"""

import logging
import time
from typing import Any, Dict, List, Optional

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

from .second_brain import SecondBrain

from .rate_limited_fetcher import RateLimitedFetcher

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class RSSIngester:
    """
    Ingests RSS/Atom feeds into the Second Brain with tagging.
    """

    def __init__(self, second_brain: SecondBrain, calls_per_minute: int = 30):
        self.second_brain = second_brain
        self.fetcher = RateLimitedFetcher(calls_per_minute=calls_per_minute)

        if not HAS_FEEDPARSER:
            logging.warning("feedparser not installed. RSSIngester will be limited.")

    def ingest_feed(self, url: str, tags: Optional[List[str]] = None) -> int:
        if not HAS_FEEDPARSER:
            logging.error("feedparser is required for RSS ingestion")
            return 0

        tags = tags or ["rss", "external"]

        def fetch():
            return feedparser.parse(url)

        feed = self.fetcher.fetch(fetch)

        if feed.bozo:
            logging.warning(f"Failed to parse feed: {url}")
            return 0

        count = 0
        for entry in feed.entries:
            finding = {
                "title": entry.get("title", "Untitled"),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", "")[:2000],
                "published": entry.get("published", ""),
                "tags": tags + [tag.term for tag in entry.get("tags", [])],
                "source": "rss",
            }
            self.second_brain.store_finding(finding)
            count += 1

        logging.info(f"Ingested {count} items from {url}")
        return count
