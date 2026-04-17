"""src/serving/cli.py

Three decomposed CLI commands: retrieve, rerank, explain.
ADR-005: CLI over FastAPI — RecSys signal > HTTP plumbing.
         Each command emits schema-validated JSON (Pydantic models).
         Decomposition makes the two-stage pipeline explicit and enables
         per-stage latency profiling.

Usage:
    python -m src.serving.cli retrieve --query "chill rainy afternoon jazz" \\
        --top-k 50 --index-tier auto --json-output

    python -m src.serving.cli rerank \\
        --query "chill rainy afternoon jazz" \\
        --candidates candidates.json --top-k 10 --json-output

    python -m src.serving.cli explain \\
        --query "chill rainy afternoon jazz" \\
        --tracks tracks.json --json-output

Import discipline (CLAUDE.md): serving/ imports from models/ and data/schema.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from src.data.schema import (
    AppConfig,
    CandidateTrack,
    ExplainRequest,
    RerankRequest,
    RetrieveRequest,
)
from src.models.encoder import Encoder
from src.models.reranker import Reranker
from src.serving.explainer import Explainer
from src.serving.ranker import Ranker
from src.serving.retriever import Retriever

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> AppConfig:
    """Load and validate AppConfig from config.yaml.

    ADR-006: Pydantic validation at startup — any missing/misconfigured key
    is a hard failure, not a silent runtime error.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def build_components(config: AppConfig) -> tuple[Retriever, Ranker, Explainer]:
    """Initialize all serving components from config.

    ADR-006: Components initialized once; stateless between requests.
    Instantiation order: Encoder → Retriever → Reranker → Ranker → Explainer.
    """
    encoder = Encoder(config)
    retriever = Retriever(config, encoder)
    reranker = Reranker(config)
    ranker = Ranker(config, retriever, reranker)
    explainer = Explainer(config)
    return retriever, ranker, explainer


# ---------------------------------------------------------------------------
# retrieve command
# ADR-005: Stage 1 — dense + BM25 hybrid ANN retrieval
# ---------------------------------------------------------------------------

def cmd_retrieve(args: argparse.Namespace, config: AppConfig) -> None:
    """Execute the retrieve command.

    Validates input via RetrieveRequest (failure_modes.md §4.4), then
    runs Retriever.retrieve() and prints JSON output.
    """
    request = RetrieveRequest(
        query=args.query,
        top_k=args.top_k,
        index_tier=args.index_tier,
    )
    retriever, _, _ = build_components(config)
    response = retriever.retrieve(request)

    if args.json_output:
        print(response.model_dump_json(indent=2))
    else:
        print(f"Query: {response.query}")
        print(f"Index tier: {response.index_tier_used} | Retrieval: {response.retrieval_ms:.1f}ms")
        if response.warning:
            print(f"Warning: {response.warning}")
        for i, track in enumerate(response.candidates, 1):
            score = f"{track.rrf_score:.3f}" if track.rrf_score is not None else "N/A"
            print(f"  {i:2}. {track.title} — {track.artist} [{track.genre_top}] (rrf: {score})")


# ---------------------------------------------------------------------------
# rerank command
# ADR-005: Stage 2 — cross-encoder reranking
# ---------------------------------------------------------------------------

def cmd_rerank(args: argparse.Namespace, config: AppConfig) -> None:
    """Execute the rerank command.

    Reads candidates from --candidates JSON file, validates via RerankRequest,
    runs Reranker.rerank(), prints JSON output.
    """
    raw = json.loads(Path(args.candidates).read_text())
    # Accept either a list of CandidateTrack dicts or a full RetrieveResponse dict.
    if isinstance(raw, list):
        candidates = [CandidateTrack.model_validate(c) for c in raw]
    else:
        candidates = [CandidateTrack.model_validate(c) for c in raw["candidates"]]

    request = RerankRequest(
        query=args.query,
        candidates=candidates,
        top_k=args.top_k,
    )
    _, ranker, _ = build_components(config)
    response = ranker.rerank(request)

    if args.json_output:
        print(response.model_dump_json(indent=2))
    else:
        print(f"Query: {response.query}")
        print(f"Reranker used: {response.reranker_used} | Rerank: {response.rerank_ms:.1f}ms")
        if response.warning:
            print(f"Warning: {response.warning}")
        for i, track in enumerate(response.tracks, 1):
            score = f"{track.rerank_score:.3f}" if track.rerank_score is not None else "N/A"
            print(f"  {i:2}. {track.title} — {track.artist} [{track.genre_top}] (rerank: {score})")


# ---------------------------------------------------------------------------
# explain command
# ADR-005: Async/offline — not on critical retrieval path
# ---------------------------------------------------------------------------

def cmd_explain(args: argparse.Namespace, config: AppConfig) -> None:
    """Execute the explain command.

    Reads tracks from --tracks JSON file, validates via ExplainRequest,
    runs Explainer.explain(), prints JSON output.
    """
    raw = json.loads(Path(args.tracks).read_text())
    # Accept either a list of CandidateTrack dicts or a full RerankResponse dict.
    if isinstance(raw, list):
        tracks = [CandidateTrack.model_validate(t) for t in raw]
    else:
        tracks = [CandidateTrack.model_validate(t) for t in raw["tracks"]]

    request = ExplainRequest(query=args.query, tracks=tracks)
    _, _, explainer = build_components(config)
    response = explainer.explain(request)

    if args.json_output:
        print(response.model_dump_json(indent=2))
    else:
        print(f"Query: {response.query}")
        print(f"Explain: {response.explain_ms:.1f}ms")
        for i, track in enumerate(response.tracks, 1):
            explanation = track.explanation or "[no explanation]"
            print(f"  {i:2}. {track.title} — {track.artist}")
            print(f"      {explanation}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.serving.cli",
        description="NLP Music Playlist Retrieval — serving CLI (ADR-005)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # retrieve
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Stage 1: Dense + BM25 hybrid ANN retrieval",
    )
    retrieve_parser.add_argument("--query",      required=True,  help="Free-text query")
    retrieve_parser.add_argument("--top-k",      type=int, default=50)
    retrieve_parser.add_argument("--index-tier", choices=["auto", "full", "metadata_only"], default="auto")
    retrieve_parser.add_argument("--json-output", action="store_true")

    # rerank
    rerank_parser = subparsers.add_parser(
        "rerank",
        help="Stage 2: Cross-encoder reranking of retrieved candidates",
    )
    rerank_parser.add_argument("--query",      required=True)
    rerank_parser.add_argument("--candidates", required=True, help="Path to candidates JSON from retrieve")
    rerank_parser.add_argument("--top-k",      type=int, default=10)
    rerank_parser.add_argument("--json-output", action="store_true")

    # explain
    explain_parser = subparsers.add_parser(
        "explain",
        help="Per-track explanation via Mistral 7B (offline/async)",
    )
    explain_parser.add_argument("--query",       required=True)
    explain_parser.add_argument("--tracks",      required=True, help="Path to tracks JSON from rerank")
    explain_parser.add_argument("--json-output", action="store_true")

    return parser


def main() -> None:
    """Entry point: python -m src.serving.cli <command> [args]."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "retrieve":
        cmd_retrieve(args, config)
    elif args.command == "rerank":
        cmd_rerank(args, config)
    elif args.command == "explain":
        cmd_explain(args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
