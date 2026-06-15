"""tests/data/test_ingest.py

Tests for GeniusClient — the guarded multi-result lyrics fallback, the per-hit
artist/title match guard, and 429 rate-limit backoff (ADR-002 amendment
2026-06-12). httpx is mocked so no network is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data import ingest
from src.data.ingest import GeniusClient
from src.data.schema import AppConfig

LYRICS_HTML = '<div data-lyrics-container="true">Walking in the rain<br/>feeling alone tonight</div>'
EMPTY_HTML = "<html><body>no lyrics container here</body></html>"


@pytest.fixture(autouse=True)
def _genius_token(monkeypatch):
    """GeniusClient.__init__ requires a token (read as a module global)."""
    monkeypatch.setattr(ingest, "GENIUS_TOKEN", "test-token")


def _resp(status=200, json_data=None, text="", retry_after=None):
    r = MagicMock()
    r.status_code = status
    r.headers = {"Retry-After": retry_after} if retry_after is not None else {}
    r.json = MagicMock(return_value=json_data or {})
    r.text = text
    if status >= 400 and status != 429:
        r.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=r)
        )
    else:
        r.raise_for_status = MagicMock()
    return r


def _hit(title, artist, url):
    return {"result": {"title": title, "url": url, "primary_artist": {"name": artist}}}


def _search(*hits):
    return _resp(json_data={"response": {"hits": list(hits)}})


def _client(*responses):
    c = MagicMock()
    c.get = AsyncMock(side_effect=list(responses))
    return c


class TestMatchGuard:
    def test_accepts_matching_hit(self):
        client = GeniusClient(AppConfig())
        hit = _hit("Rainy Day Blues", "Test Artist A", "http://x")
        assert client._validate_genius_hit(hit, "Test Artist A", "Rainy Day Blues") is True

    def test_rejects_wrong_artist(self):
        client = GeniusClient(AppConfig())
        hit = _hit("Rainy Day Blues", "Completely Different Band", "http://x")
        assert client._validate_genius_hit(hit, "Test Artist A", "Rainy Day Blues") is False

    def test_rejects_wrong_title(self):
        client = GeniusClient(AppConfig())
        hit = _hit("Totally Unrelated Song Name", "Test Artist A", "http://x")
        assert client._validate_genius_hit(hit, "Test Artist A", "Rainy Day Blues") is False


class TestFallback:
    @pytest.mark.asyncio
    async def test_falls_through_empty_first_hit(self):
        """hit[0] has no lyrics container → advance to hit[1] which does."""
        client = GeniusClient(AppConfig())
        search = _search(
            _hit("Rainy Day Blues", "Test Artist A", "http://x/0"),
            _hit("Rainy Day Blues", "Test Artist A", "http://x/1"),
        )
        http = _client(search, _resp(text=EMPTY_HTML), _resp(text=LYRICS_HTML))
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is not None and "Walking in the rain" in lyrics
        assert http.get.call_count == 3  # search + both pages

    @pytest.mark.asyncio
    async def test_match_guard_skips_wrong_hit_without_fetching(self):
        """A wrong-artist hit is rejected by the guard before its page is fetched."""
        client = GeniusClient(AppConfig())
        search = _search(
            _hit("Rainy Day Blues", "Wrong Band Entirely", "http://x/0"),  # rejected
            _hit("Rainy Day Blues", "Test Artist A", "http://x/1"),        # accepted
        )
        http = _client(search, _resp(text=LYRICS_HTML))
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is not None and "Walking in the rain" in lyrics
        assert http.get.call_count == 2  # search + ONLY the accepted hit's page

    @pytest.mark.asyncio
    async def test_all_hits_fail_returns_none(self):
        client = GeniusClient(AppConfig())
        search = _search(
            _hit("Rainy Day Blues", "Test Artist A", "http://x/0"),
            _hit("Rainy Day Blues", "Test Artist A", "http://x/1"),
            _hit("Rainy Day Blues", "Test Artist A", "http://x/2"),
        )
        # all three pages empty
        http = _client(search, _resp(text=EMPTY_HTML), _resp(text=EMPTY_HTML), _resp(text=EMPTY_HTML))
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is None

    @pytest.mark.asyncio
    async def test_respects_max_results_limit(self):
        """With genius_max_results=2, only the first 2 hits are tried."""
        cfg = AppConfig()
        cfg.pipeline.genius_max_results = 2
        client = GeniusClient(cfg)
        search = _search(
            _hit("Rainy Day Blues", "Test Artist A", "http://x/0"),
            _hit("Rainy Day Blues", "Test Artist A", "http://x/1"),
            _hit("Rainy Day Blues", "Test Artist A", "http://x/2"),  # never tried
        )
        http = _client(search, _resp(text=EMPTY_HTML), _resp(text=EMPTY_HTML))
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is None
        assert http.get.call_count == 3  # search + 2 pages (not 3)


class TestRateLimitBackoff:
    @pytest.mark.asyncio
    async def test_429_retries_same_request(self):
        """429 → backoff and retry the SAME request, not advance to next hit."""
        client = GeniusClient(AppConfig())
        search = _search(_hit("Rainy Day Blues", "Test Artist A", "http://x/0"))
        # search: 429 (Retry-After 0 → no real wait) then OK; then page lyrics
        http = _client(_resp(status=429, retry_after="0"), search, _resp(text=LYRICS_HTML))
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is not None and "Walking in the rain" in lyrics
        assert http.get.call_count == 3  # 429 retry + search-ok + page

    @pytest.mark.asyncio
    async def test_429_exhausts_retries_returns_none(self):
        cfg = AppConfig()
        cfg.enrichment.max_retries = 2
        client = GeniusClient(cfg)
        http = _client(_resp(status=429, retry_after="0"), _resp(status=429, retry_after="0"))
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is None
        assert http.get.call_count == 2  # both attempts 429, then give up

    @pytest.mark.asyncio
    async def test_429_long_retry_after_skips_immediately(self):
        """A Retry-After beyond the cap → skip the track, don't sleep for an hour."""
        cfg = AppConfig()
        cfg.pipeline.genius_retry_after_cap_s = 60.0
        client = GeniusClient(cfg)
        http = _client(_resp(status=429, retry_after="3600"))  # 1 hour > 60s cap
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is None
        assert http.get.call_count == 1  # gave up on the first 429, no retry/sleep

    @pytest.mark.asyncio
    async def test_429_quota_exhausted_raises(self):
        """remaining=0 is the daily quota being gone (not a short window) → raise
        GeniusQuotaExceeded so the crawl aborts cleanly instead of skip-flooding."""
        client = GeniusClient(AppConfig())
        r = _resp(status=429)
        r.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "42464"}
        http = _client(r)
        with pytest.raises(ingest.GeniusQuotaExceeded) as exc:
            await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert exc.value.reset_s == 42464.0
        assert http.get.call_count == 1  # no pointless retries against a blown quota

    @pytest.mark.asyncio
    async def test_quota_exhausted_propagates_through_batch(self):
        """batch_get_lyrics must NOT swallow GeniusQuotaExceeded — it aborts the
        whole batch so build_catalog can checkpoint and stop."""
        client = GeniusClient(AppConfig())
        r = _resp(status=429)
        r.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "42464"}
        client.get_lyrics = AsyncMock(side_effect=ingest.GeniusQuotaExceeded(42464.0))
        with pytest.raises(ingest.GeniusQuotaExceeded):
            await client.batch_get_lyrics([("Artist A", "Song A")], concurrency=2)

    @pytest.mark.asyncio
    async def test_429_no_retry_after_uses_reset_header(self, monkeypatch):
        """When Retry-After is absent (Genius's actual behavior), the wait falls back
        to x-ratelimit-reset; a reset beyond the cap skips rather than 1/2/4s spinning."""
        cfg = AppConfig()
        cfg.pipeline.genius_retry_after_cap_s = 900.0
        client = GeniusClient(cfg)
        r = _resp(status=429)  # no Retry-After, remaining>0 (short throttle), big reset
        r.headers = {"x-ratelimit-remaining": "5", "x-ratelimit-reset": "3600"}
        http = _client(r)
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is None
        assert http.get.call_count == 1  # 3600 > 900 cap → skip immediately, no spin

    @pytest.mark.asyncio
    async def test_429_within_cap_waits_then_retries_no_skip(self, monkeypatch):
        """A normal ~8min window (under cap) pauses the crawl and retries — the
        track is recovered, NOT skipped (the bug that was losing lyrics)."""
        cfg = AppConfig()
        cfg.pipeline.genius_retry_after_cap_s = 900.0  # 15 min
        client = GeniusClient(cfg)

        # Fake clock that jumps far forward each call so the shared pause elapses
        # immediately; no-op sleep so no real wall-clock wait.
        clock = {"t": 0.0}
        monkeypatch.setattr(ingest.time, "monotonic", lambda: clock.__setitem__("t", clock["t"] + 1000.0) or clock["t"])
        monkeypatch.setattr(ingest.asyncio, "sleep", AsyncMock())

        search = _search(_hit("Rainy Day Blues", "Test Artist A", "http://x/0"))
        http = _client(_resp(status=429, retry_after="468"), search, _resp(text=LYRICS_HTML))
        lyrics = await client.get_lyrics("Test Artist A", "Rainy Day Blues", client=http)
        assert lyrics is not None and "Walking in the rain" in lyrics
        assert http.get.call_count == 3  # 429 (paused) + search-ok + page — recovered
