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
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd

from src.data.schema import AppConfig

logger = logging.getLogger(__name__)

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

    async def get_lyrics(self, artist: str, title: str) -> str | None:
        """Fetch raw lyrics for a track. Returns None on miss or API error.

        Does NOT clean lyrics — cleaning is the responsibility of preprocess.py.
        Two-step: search the Genius API for the track URL, then scrape the page.
        """
        headers = {"Authorization": f"Bearer {GENIUS_TOKEN}"}

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Step 1: Search for the track via the Genius API
            try:
                search_response = await client.get(
                    f"{self._BASE_URL}/search",
                    headers=headers,
                    params={"q": f"{title} {artist}"},
                )
                search_response.raise_for_status()
            except httpx.TimeoutException:
                logger.warning("Genius search timeout for '%s' by '%s'", title, artist)
                return None
            except httpx.HTTPStatusError as e:
                logger.warning("Genius search HTTP error for '%s' by '%s': %s", title, artist, e)
                return None

            hits = search_response.json().get("response", {}).get("hits", [])
            if not hits:
                logger.debug("No Genius results for '%s' by '%s'", title, artist)
                return None

            lyrics_url = hits[0]["result"]["url"]

            # Step 2: Scrape the lyrics page
            try:
                page_response = await client.get(lyrics_url)
                page_response.raise_for_status()
            except httpx.TimeoutException:
                logger.warning("Genius page timeout for %s", lyrics_url)
                return None
            except httpx.HTTPStatusError as e:
                logger.warning("Genius page HTTP error for %s: %s", lyrics_url, e)
                return None

            return self._extract_lyrics(page_response.text)

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

        async def fetch_one(artist: str, title: str) -> tuple[str, str | None]:
            key = f"{artist}|{title}"
            async with semaphore:
                try:
                    lyrics = await self.get_lyrics(artist, title)
                    return key, lyrics
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
