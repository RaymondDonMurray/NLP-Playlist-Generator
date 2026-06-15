"""src/data/ingest.py

API clients for the three data sources: FMA, Spotify, Genius.
ADR-002: Batch pipeline. Sources combined in build_catalog.py entry point.
scope.md §Basic Data Pipeline: FMA direct download, Spotify via spotipy,
Genius via async crawler.
"""

from __future__ import annotations

import asyncio
import html as html_module
import logging
import re
import time
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd
from rapidfuzz import fuzz

from src.data.schema import AppConfig

logger = logging.getLogger(__name__)


class GeniusQuotaExceeded(RuntimeError):
    """The Genius *daily* request quota is exhausted (x-ratelimit-remaining=0).

    Distinct from a short in-window throttle: nothing recovers this until the
    daily counter resets (hours away), so the crawl aborts cleanly — progress is
    checkpointed by the caller — rather than skip-flooding every remaining track
    and burning further requests against an already-blown limit.
    """

    def __init__(self, reset_s: float | None = None) -> None:
        self.reset_s = reset_s
        msg = "Genius daily quota exhausted"
        if reset_s:
            msg += f"; resets in ~{reset_s / 3600:.1f}h"
        super().__init__(msg)


import os
from dotenv import load_dotenv

load_dotenv()
GENIUS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN", None)
SPOTIFY_ID = os.environ.get("SPOTIFY_CLIENT_ID", None)
SPOTIFY_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", None)


# ---------------------------------------------------------------------------
# FMA (Free Music Archive)
# scope.md: Direct download fma_metadata (~342MB), github.com/mdeff/fma
# ---------------------------------------------------------------------------

class FMAClient:
    """Loads FMA metadata from locally downloaded fma_metadata directory."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.paths.raw_dir)

    def load_tracks(self) -> Iterator[dict]:
        """Yield raw FMA track dicts from fma_metadata/tracks.csv.

        The FMA CSV uses a two-row multi-level header (category, field).
        Yields one dict per English-language track with essential fields
        present. Skips tracks with missing title or artist.

        Raises FileNotFoundError if fma_metadata has not been downloaded.
        """
        tracks_path = self.raw_dir / "fma_metadata" / "tracks.csv"
        if not tracks_path.exists():
            raise FileNotFoundError(
                f"FMA tracks.csv not found at {tracks_path}. "
                "Download fma_metadata from https://github.com/mdeff/fma"
            )

        # FMA uses a two-level header: (category, field)
        tracks = pd.read_csv(str(tracks_path), index_col=0, header=[0, 1])

        n_total = 0
        n_skipped_lang = 0
        n_skipped_missing = 0

        for track_id, row in tracks.iterrows():
            n_total += 1
            try:
                title = row[("track", "title")]
                artist = row[("artist", "name")]
                genre_top = row[("track", "genre_top")]
                duration_s = row[("track", "duration")]
                language_code = row[("track", "language_code")]
            except KeyError as e:
                logger.warning("Missing expected column in FMA row %s: %s", track_id, e)
                n_skipped_missing += 1
                continue

            if pd.isna(title) or pd.isna(artist):
                n_skipped_missing += 1
                continue

            # Popularity signal — read non-fatally; absence means 0, never a skip.
            listens = row.get(("track", "listens"))

            # scope.md: Restrict to English for MVP
            if pd.notna(language_code) and str(language_code) != "en":
                n_skipped_lang += 1
                continue

            yield {
                "track_id": f"fma_{int(str(track_id)):06d}",
                "title": str(title).strip(),
                "artist": str(artist).strip(),
                "genre_top": str(genre_top).strip() if pd.notna(genre_top) else "Unknown",
                # FMA stores duration in seconds; schema expects milliseconds
                "duration_ms": int(float(duration_s) * 1000) if pd.notna(duration_s) else None,
                "listens": int(listens) if pd.notna(listens) else 0,
            }

        logger.info(
            "FMA load_tracks: %d total, %d skipped (non-English), %d skipped (missing fields)",
            n_total, n_skipped_lang, n_skipped_missing,
        )

    def load_audio_features(self) -> pd.DataFrame:
        """Load EchoNest audio features from echonest.csv.

        Returns a DataFrame indexed by integer track_id with columns:
        acousticness, danceability, energy, instrumentalness, liveness,
        speechiness, tempo, valence.

        FMA's echonest.csv covers ~13K tracks and provides the same audio
        features as Spotify's audio-features API, which is now restricted (403)
        for new apps created after November 2024.

        Raises FileNotFoundError if echonest.csv is not present.
        """
        echonest_path = self.raw_dir / "fma_metadata" / "echonest.csv"
        if not echonest_path.exists():
            raise FileNotFoundError(f"FMA echonest.csv not found at {echonest_path}")

        ec = pd.read_csv(str(echonest_path), index_col=0, header=[0, 1, 2])
        features = ec[("echonest", "audio_features")].copy()
        features.index.name = "track_id"
        logger.info("Loaded EchoNest audio features for %d tracks", len(features))
        return features

    def load_url_to_track_id(self) -> dict[str, int]:
        """Return mapping of FMA track_url → integer track_id from raw_tracks.csv.

        The HuggingFace `free-music-archive-large` dataset carries no track_id;
        the `url` field is the deterministic join key back to FMA (exact match,
        ~99% coverage). Rows with a null or duplicate url are skipped — callers
        fall back to the zero-padded audio filename, else drop the track
        (ADR-002 amendment 2026-06-12).

        Raises FileNotFoundError if raw_tracks.csv is not present.
        """
        raw_tracks_path = self.raw_dir / "fma_metadata" / "raw_tracks.csv"
        if not raw_tracks_path.exists():
            raise FileNotFoundError(f"FMA raw_tracks.csv not found at {raw_tracks_path}")

        rt = pd.read_csv(str(raw_tracks_path), index_col=0, usecols=["track_id", "track_url"])
        url_map: dict[str, int] = {}
        seen_dupes: set[str] = set()
        for track_id, url in rt["track_url"].items():
            if pd.isna(url):
                continue
            url = str(url)
            if url in url_map:
                # Collision: drop both — an ambiguous url cannot be a safe join key.
                seen_dupes.add(url)
                continue
            url_map[url] = int(track_id)
        for dup in seen_dupes:
            url_map.pop(dup, None)

        logger.info(
            "Built url→track_id map: %d unique urls (%d dropped as duplicate/null)",
            len(url_map), len(rt) - len(url_map),
        )
        return url_map

    def load_genres(self) -> dict[int, str]:
        """Return mapping of genre_id → genre_name from genres.csv."""
        genres_path = self.raw_dir / "fma_metadata" / "genres.csv"
        if not genres_path.exists():
            raise FileNotFoundError(f"FMA genres.csv not found at {genres_path}")

        genres_df = pd.read_csv(str(genres_path))
        return dict(zip(genres_df["genre_id"], genres_df["title"]))


# ---------------------------------------------------------------------------
# Spotify Web API
# scope.md: spotipy, ~2 min for 5K tracks at free tier, fuzzy-join to FMA
# ---------------------------------------------------------------------------

class SpotifyClient:
    """Fetches audio features from Spotify Web API via spotipy.

    Credentials loaded from environment: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET.
    ADR-006: API keys in .env only, never in config.yaml.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = None   # initialized lazily to avoid import-time auth

    def _get_client(self):
        """Lazy-init spotipy client from environment credentials."""
        if self._client is not None:
            return self._client

        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials

        if not SPOTIFY_ID or not SPOTIFY_SECRET:
            raise EnvironmentError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env"
            )

        self._client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_ID,
                client_secret=SPOTIFY_SECRET,
            )
        )
        logger.info("Spotify client initialized")
        return self._client

    def get_audio_features(self, spotify_ids: list[str]) -> list[dict]:
        """Batch-fetch audio features for up to 100 track IDs per request.

        Returns list of raw Spotify audio feature dicts.
        Handles rate limiting via spotipy's built-in retry.
        Skips None responses (tracks Spotify failed to analyze).
        """
        sp = self._get_client()
        results: list[dict] = []

        for i in range(0, len(spotify_ids), 100):
            batch = spotify_ids[i : i + 100]
            try:
                features = sp.audio_features(batch)
            except Exception as e:
                logger.error("Spotify audio_features batch [%d:%d] failed: %s", i, i + 100, e)
                continue

            if features:
                valid = [f for f in features if f is not None]
                results.extend(valid)
                logger.debug("Fetched %d/%d audio features for batch starting at %d", len(valid), len(batch), i)

        return results

    def search_track(self, artist: str, title: str) -> dict | None:
        """Search for a single track; return first result or None.

        Returns the raw Spotify track object from the search result, which
        includes: id, name, artists, duration_ms, and a nested audio_features
        key (populated separately via get_audio_features).
        """
        sp = self._get_client()
        query = f"track:{title} artist:{artist}"
        try:
            results = sp.search(q=query, type="track", limit=1)
        except Exception as e:
            logger.warning("Spotify search failed for '%s' by '%s': %s", title, artist, e)
            return None

        if results is None:
            return None
        items = results.get("tracks", {}).get("items", [])
        if not items:
            logger.debug("No Spotify results for '%s' by '%s'", title, artist)
            return None

        track = items[0]
        logger.debug(
            "Spotify match: '%s' by '%s' → '%s' by '%s' (id=%s)",
            title, artist,
            track.get("name"), track.get("artists", [{}])[0].get("name"),
            track.get("id"),
        )
        return track


# ---------------------------------------------------------------------------
# Genius API
# scope.md: Async crawler, ~5hr crawl for 5K tracks, ~60-70% coverage
# ---------------------------------------------------------------------------

class GeniusClient:
    """Fetches lyrics from Genius API.

    Credentials loaded from environment: GENIUS_ACCESS_TOKEN.
    ADR-006: API keys in .env only.
    failure_modes.md §1.2: Raw lyrics passed through clean_lyrics() in preprocess.py.
    """

    _BASE_URL = "https://api.genius.com"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        if not GENIUS_TOKEN:
            raise EnvironmentError("GENIUS_ACCESS_TOKEN must be set in .env")
        # Shared rate-limit gate: a single 429 pauses the WHOLE crawl until the
        # window clears, instead of each concurrent task independently skipping
        # its track. monotonic deadline; 0.0 = not paused.
        self._paused_until: float = 0.0

    async def get_lyrics(
        self, artist: str, title: str, client: httpx.AsyncClient | None = None
    ) -> str | None:
        """Fetch raw lyrics for a track. Returns None on miss or API error.

        Does NOT clean lyrics — cleaning is preprocess.py's job. Searches Genius,
        then walks up to `pipeline.genius_max_results` hits, accepting the first
        that BOTH passes the artist/title match guard AND has lyrics. Reuses an
        injected client when given (batch crawl pools one connection).
        """
        if client is None:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as own:
                return await self._get_lyrics(artist, title, own)
        return await self._get_lyrics(artist, title, client)

    async def _get_lyrics(
        self, artist: str, title: str, client: httpx.AsyncClient
    ) -> str | None:
        headers = {"Authorization": f"Bearer {GENIUS_TOKEN}"}

        # Step 1: search
        search_response = await self._get_with_backoff(
            client, f"{self._BASE_URL}/search", headers=headers, params={"q": f"{title} {artist}"}
        )
        if search_response is None:
            return None
        hits = search_response.json().get("response", {}).get("hits", [])
        if not hits:
            logger.debug("No Genius results for '%s' by '%s'", title, artist)
            return None

        # Step 2: walk hits with a per-hit match guard (failure_modes.md §1.1 — a
        # blind fall-through to lower-ranked hits risks pulling a wrong song's lyrics).
        max_results = self.config.pipeline.genius_max_results
        for rank, hit in enumerate(hits[:max_results]):
            if not self._validate_genius_hit(hit, artist, title):
                continue
            url = (hit.get("result") or {}).get("url")
            if not url:
                continue
            page = await self._get_with_backoff(client, url, headers=headers)
            if page is None:
                continue   # timeout / 404 on this hit → try the next one
            lyrics = self._extract_lyrics(page.text)
            if lyrics:
                if rank > 0:
                    logger.info(
                        "Genius fallback: rank %d hit used for '%s' by '%s'", rank, title, artist
                    )
                return lyrics

        logger.debug("No usable Genius lyrics in top-%d for '%s' by '%s'", max_results, title, artist)
        return None

    def _validate_genius_hit(self, hit: dict, artist: str, title: str) -> bool:
        """Guard: the Genius hit must fuzzily match the FMA (artist, title).

        Prevents a fallback to lower-ranked results from silently grabbing a
        different song's lyrics (the lyrics analog of failure_modes.md §1.1).
        Reuses the catalog fuzzy thresholds.
        """
        result = hit.get("result") or {}
        g_title = result.get("title") or ""
        g_artist = (result.get("primary_artist") or {}).get("name") or ""
        title_sim = fuzz.token_sort_ratio(title, g_title)
        artist_sim = fuzz.token_sort_ratio(artist, g_artist)
        ok = (
            title_sim >= self.config.pipeline.fuzzy_match_threshold
            and artist_sim >= self.config.pipeline.artist_match_threshold
        )
        if not ok:
            logger.debug(
                "Rejected Genius hit '%s' by '%s' (title=%.0f artist=%.0f) for '%s'/'%s'",
                g_title, g_artist, title_sim, artist_sim, title, artist,
            )
        return ok

    async def _respect_rate_limit(self) -> None:
        """Block until any crawl-wide rate-limit pause has elapsed.

        Called before every request so that one task's 429 throttles ALL tasks,
        not just itself. Sleeps in short slices so a pause extended by another
        task mid-wait is picked up promptly.
        """
        while True:
            remaining = self._paused_until - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 5.0))

    def _pause_crawl(self, seconds: float) -> None:
        """Extend the shared pause deadline so all tasks wait out this window once."""
        self._paused_until = max(self._paused_until, time.monotonic() + seconds)

    async def _get_with_backoff(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        headers: dict | None = None,
        params: dict | None = None,
    ) -> httpx.Response | None:
        """GET with crawl-wide rate-limit backoff. 429 → pause the whole crawl
        (honoring Retry-After) and retry the SAME request; timeout / other HTTP
        error → give up (caller advances to the next hit).

        The 429-vs-other split matters: treating a 429 like an empty result and
        falling through to more requests makes the throttling worse. Genius's
        rate-limit window (~8 min) is a NORMAL recoverable throttle — we pause
        the entire crawl on a shared timer and wait it out so no lyrics are lost.
        Only a Retry-After beyond genius_retry_after_cap_s (a pathological,
        multi-window ban) is treated as "give up on this track".
        """
        max_retries = self.config.enrichment.max_retries
        cap = self.config.pipeline.genius_retry_after_cap_s
        for attempt in range(max_retries):
            await self._respect_rate_limit()  # honor any crawl-wide pause first
            try:
                resp = await client.get(url, headers=headers, params=params)
            except httpx.TimeoutException:
                logger.warning("Genius request timeout for %s", url)
                return None

            if resp.status_code == 429:
                # Daily quota gone (account-wide): no per-track skip helps — abort the
                # whole crawl so the caller can checkpoint and resume after the reset.
                if resp.headers.get("x-ratelimit-remaining") == "0":
                    reset = resp.headers.get("x-ratelimit-reset")
                    reset_s = float(reset) if reset and reset.isdigit() else None
                    raise GeniusQuotaExceeded(reset_s)
                # Short in-window throttle. Genius often omits Retry-After here, so fall
                # back to x-ratelimit-reset, then to exponential backoff.
                retry_after = resp.headers.get("Retry-After")
                reset = resp.headers.get("x-ratelimit-reset")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                elif reset and reset.isdigit():
                    wait = float(reset)
                else:
                    wait = float(min(2 ** attempt, cap))
                if wait > cap:
                    logger.warning(
                        "Genius 429 on %s wants %.0fs (> cap %.0fs) — pathological ban, skipping track",
                        url, wait, cap,
                    )
                    return None
                self._pause_crawl(wait)
                logger.warning(
                    "Genius 429 — pausing crawl %.0fs to wait out the rate-limit window "
                    "(attempt %d/%d)",
                    wait, attempt + 1, max_retries,
                )
                continue  # _respect_rate_limit at the top of the loop does the sleeping

            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.warning("Genius HTTP error for %s: %s", url, e)
                return None
            return resp

        logger.warning("Genius exhausted %d retries (429) for %s", max_retries, url)
        return None

    def _extract_lyrics(self, html: str) -> str | None:
        """Extract raw lyrics text from a Genius HTML page.

        Genius stores lyrics in <div data-lyrics-container="true"> elements.
        Uses regex rather than BeautifulSoup to avoid the dependency.
        Returns None if no lyrics container is found (e.g., instrumentals).
        """
        # Find all data-lyrics-container divs — may be multiple for multi-section songs
        containers = re.findall(
            r'<div[^>]*data-lyrics-container="true"[^>]*>(.*?)</div>',
            html,
            re.DOTALL,
        )
        if not containers:
            return None

        text = "\n".join(containers)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)                # strip all remaining tags
        text = html_module.unescape(text)                  # &amp; → &, etc.
        text = re.sub(r"[ \t]+", " ", text)                # collapse horizontal whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)             # collapse blank lines
        text = text.strip()

        return text or None

    async def batch_get_lyrics(
        self,
        tracks: list[tuple[str, str]],
        concurrency: int = 5,
    ) -> dict[str, str | None]:
        """Fetch lyrics for a list of (artist, title) tuples concurrently.

        Returns mapping of "{artist}|{title}" → raw lyrics or None.
        scope.md: ~5hr crawl for 5K tracks at free tier concurrency.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            async def fetch_one(artist: str, title: str) -> tuple[str, str | None]:
                key = f"{artist}|{title}"
                async with semaphore:
                    try:
                        lyrics = await self.get_lyrics(artist, title, client=client)
                        return key, lyrics
                    except GeniusQuotaExceeded:
                        raise  # daily quota gone — abort the whole batch, don't swallow
                    except Exception as e:
                        logger.warning(
                            "Unexpected error fetching lyrics for '%s' by '%s': %s",
                            title, artist, e,
                        )
                        return key, None

            tasks = [fetch_one(artist, title) for artist, title in tracks]
            results = await asyncio.gather(*tasks)

        hit_count = sum(1 for _, v in results if v is not None)
        logger.info(
            "batch_get_lyrics: %d/%d tracks with lyrics (%.0f%% coverage)",
            hit_count, len(tracks), 100 * hit_count / len(tracks) if tracks else 0,
        )
        return dict(results)
