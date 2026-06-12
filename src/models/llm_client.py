"""src/models/llm_client.py

Backend-selectable LLM transport for descriptor generation and per-track
explanation. Owns the OpenAI-compatible request shape and the shared
retry/timeout loop so descriptor.py and explainer.py never hand-roll HTTP.

Backends (config.enrichment.llm_backend):
- "lmstudio": OpenAI-compatible server (POST /v1/chat/completions). Default for
  local Apple-Silicon dev — runs MLX-quantised models (ADR-001 Mistral 7B).
- "ollama":   native Ollama API (POST /api/generate). Used by the Docker/Linux
  sidecar where MLX is unavailable.

failure_modes.md §4.3: Hard timeout per request; exponential backoff on
transient failures; fail fast on a dead process; never retry schema errors
(schema validation happens in the caller, not here).

Import discipline (CLAUDE.md): models/ imports from data/schema.py only.
"""

from __future__ import annotations

import logging
import time

import httpx

from src.data.schema import AppConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin transport wrapper around the configured local LLM backend.

    Stateless apart from config-derived connection settings. The caller owns
    the httpx.Client lifecycle (so a batch can reuse one pooled connection) and
    passes it into each call.
    """

    def __init__(self, config: AppConfig) -> None:
        enr = config.enrichment
        self.backend = enr.llm_backend
        self.base_url = enr.llm_base_url.rstrip("/")
        self.model = enr.llm_model
        self.timeout = enr.llm_timeout_s
        self.max_retries = enr.max_retries

    # ------------------------------------------------------------------ URLs
    def _chat_url(self) -> str:
        if self.backend == "ollama":
            return f"{self.base_url}/api/generate"
        return f"{self.base_url}/chat/completions"

    def _health_url(self) -> str:
        if self.backend == "ollama":
            return f"{self.base_url}/api/tags"
        return f"{self.base_url}/models"

    def _connect_error_msg(self) -> str:
        if self.backend == "ollama":
            return (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Start Ollama with: ollama serve"
            )
        return (
            f"Cannot connect to LM Studio at {self.base_url}. "
            "Start the server with: lms server start (and load the model)."
        )

    # --------------------------------------------------------------- health
    def health_check(self) -> None:
        """Verify the backend is reachable before a batch. Raises on failure.

        Preflight guard (failure_modes.md §4.3): a 30s hang per track over 5K
        tracks is 40+ hours of blocked pipeline — fail loudly up front instead.
        """
        try:
            with httpx.Client(timeout=5.0) as probe:
                resp = probe.get(self._health_url())
                resp.raise_for_status()
            logger.info("LLM backend reachable: %s (%s)", self.backend, self.base_url)
        except httpx.ConnectError:
            raise RuntimeError(self._connect_error_msg())
        except httpx.TimeoutException:
            raise RuntimeError(
                f"LLM backend at {self.base_url} did not respond within 5s. "
                "Check that the process is healthy."
            )

    # -------------------------------------------------------------- request
    def _build_payload(self, prompt: str, json_schema: dict | None) -> dict:
        if self.backend == "ollama":
            payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
            if json_schema is not None:
                payload["format"] = "json"
            return payload

        # lmstudio (OpenAI-compatible)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if json_schema is not None:
            # strict=False: equally reliable as strict=True on MLX grammars but
            # tolerant of occasional decode quirks; Pydantic validation in the
            # caller is the real gatekeeper.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": False,
                    "schema": json_schema,
                },
            }
        return payload

    def _extract_content(self, body: dict) -> str:
        if self.backend == "ollama":
            return body["response"]
        return body["choices"][0]["message"]["content"]

    def complete(
        self,
        prompt: str,
        client: httpx.Client,
        *,
        json_schema: dict | None = None,
        context: str = "",
    ) -> str | None:
        """Send a completion request and return the raw text content.

        Returns the model's text response, or None on a non-transient failure
        (unexpected HTTP error, or retries exhausted). Raises RuntimeError if
        the backend process is unreachable (fail fast, no point retrying).

        Retry logic (failure_modes.md §4.3):
        - httpx.TimeoutException  → exponential backoff (1s, 2s, 4s), retry
        - HTTP 503                → wait 10s, retry
        - other HTTP errors       → return None immediately (no retry)
        - httpx.ConnectError      → raise RuntimeError (dead process)

        Schema validation is the caller's job: it parses the returned text and,
        on ValidationError, returns None WITHOUT retrying (the prompt/schema is
        wrong and retrying produces the same output).

        Args:
            prompt: Rendered prompt string.
            client: Caller-owned httpx.Client (reused across a batch).
            json_schema: JSON schema dict to constrain output. None → free text.
            context: Optional identifier (e.g. track_id) for log messages.
        """
        ctx = f" [{context}]" if context else ""
        payload = self._build_payload(prompt, json_schema)

        for attempt in range(self.max_retries):
            try:
                resp = client.post(self._chat_url(), json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return self._extract_content(resp.json())

            except httpx.TimeoutException:
                logger.warning(
                    "LLM timeout%s (attempt %d/%d)", ctx, attempt + 1, self.max_retries
                )
                time.sleep(2 ** attempt)  # 1s, 2s, 4s

            except httpx.ConnectError:
                raise RuntimeError(self._connect_error_msg())

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    logger.warning(
                        "LLM unavailable (503)%s — waiting 10s (attempt %d/%d)",
                        ctx, attempt + 1, self.max_retries,
                    )
                    time.sleep(10)
                else:
                    logger.error("Unexpected HTTP error%s: %s", ctx, e)
                    return None  # don't retry non-transient HTTP errors

        logger.error("Exhausted %d retries%s", self.max_retries, ctx)
        return None
