# path: src/juniorhome/rate_limited_fetcher.py
#!/usr/bin/env python3
"""
Rate Limited Fetcher

Production-grade utility for fetching real data from free APIs
while respecting rate limits. Designed to be used across the ecosystem
(JuniorHome, BitNet-mlx pipelines, JuniorAGI_SDK, etc).

Based on patterns already proven in JuniorStock.
"""

import logging
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class RateLimitedFetcher:
    """
    Fetches data from APIs while respecting rate limits.
    """

    def __init__(self, calls_per_minute: int = 60, calls_per_day: Optional[int] = None):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls: deque = deque()
        self.day_calls: deque = deque() if calls_per_day else None
        logging.info(f"RateLimitedFetcher initialized ({calls_per_minute} calls/min)")

    def _clean_old_calls(self):
        now = time.time()
        # Clean minute window
        while self.minute_calls and now - self.minute_calls[0] > 60:
            self.minute_calls.popleft()

        # Clean day window if enabled
        if self.day_calls:
            while self.day_calls and now - self.day_calls[0] > 86400:
                self.day_calls.popleft()

    def can_call(self) -> bool:
        self._clean_old_calls()
        if len(self.minute_calls) >= self.calls_per_minute:
            return False
        if self.day_calls and len(self.day_calls) >= self.calls_per_day:
            return False
        return True

    def wait_if_needed(self):
        while not self.can_call():
            time.sleep(1)

    def fetch(self, fetch_func: Callable[[], Any]) -> Any:
        self.wait_if_needed()
        self.minute_calls.append(time.time())
        if self.day_calls:
            self.day_calls.append(time.time())

        try:
            result = fetch_func()
            return result
        except Exception as e:
            logging.error(f"Fetch failed: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        self._clean_old_calls()
        return {
            "calls_last_minute": len(self.minute_calls),
            "limit_per_minute": self.calls_per_minute,
            "calls_today": len(self.day_calls) if self.day_calls else "N/A",
            "limit_per_day": self.calls_per_day or "N/A",
        }
