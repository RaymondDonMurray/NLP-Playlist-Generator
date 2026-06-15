"""scripts/check_genius_quota.py

Probe the Genius API for remaining daily request quota.

Sends a single lightweight search request and reads the rate-limit headers.
Genius only populates ``x-ratelimit-limit`` / ``x-ratelimit-remaining`` on
successful (200) responses; while throttled it returns 429 with only a
``Retry-After`` (seconds until requests are accepted again).

WARNING: each run costs one request and, while you are already throttled, may
keep the throttle window alive. Run sparingly (e.g. once an hour), never in a
loop.

Usage:
    python scripts/check_genius_quota.py
"""

from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

GENIUS_SEARCH_URL = "https://api.genius.com/search"
DAILY_LIMIT = 10_000  # Genius account-wide daily quota


def main() -> int:
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        print("GENIUS_ACCESS_TOKEN not found in .env", file=sys.stderr)
        return 1

    resp = httpx.get(
        GENIUS_SEARCH_URL,
        params={"q": "rate limit probe"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )

    print(f"status: {resp.status_code}")
    for key in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "Retry-After"):
        print(f"{key}: {resp.headers.get(key)}")

    limit = resp.headers.get("x-ratelimit-limit")
    remaining = resp.headers.get("x-ratelimit-remaining")
    if limit and remaining and limit.isdigit() and remaining.isdigit():
        print(f"\nused today: {int(limit) - int(remaining)} / {limit}")
    elif resp.status_code == 200:
        # Successful request but no headers — quota tracking unavailable.
        print(f"\nrequest succeeded; quota headers absent (daily limit is {DAILY_LIMIT})")
    else:
        print("\nstill throttled — no quota headers; see Retry-After above")

    reset = resp.headers.get("x-ratelimit-reset")
    if reset and reset.isdigit():
        print(f"resets in: ~{int(reset) / 3600:.1f}h")
    retry_after = resp.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        print(f"retry after: ~{int(retry_after) / 60:.0f} min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
