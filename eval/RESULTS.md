# Evaluation Results

**Eval set:** 10 hand-authored queries, author-labeled relevant tracks (`eval/queries.json`).
**Catalog:** ~10,000 tracks (194 `full_enrichment` with lyrics, 9,706 `metadata_only`).
**Metrics:** Precision@10, NDCG@10, MRR, Novelty@10. Retrieval uses `index_tier="auto"`.

> **Self-labeled caveat (required):** relevant tracks were judged by the author, not
> collected from user interaction data. At this scale that is standard, but it means
> the *provenance* of the labels materially affects the numbers — see
> [Labeling methodology](#labeling-methodology-and-its-effect-on-the-numbers) below.
> This is the central methodological finding of the eval, not a footnote.

---

## Ablation table

The `hybrid_rerank_expanded` config (LLM query expansion) is not implemented and is
skipped (documented in `docs/tasks.md` UQ-3), so four of the five configs are reported.

Two runs are shown deliberately. They differ **only** in how the relevant-track labels
were pooled — the code, catalog, and queries are identical. The gap between them *is*
the result.

### Run 1 — labels pooled from BM25 candidates only

Relevant tracks were surfaced with `scripts/label_candidates.py` (BM25 over descriptor
text) and hand-judged.

| Config | P@10 | NDCG@10 | MRR | Novelty@10 |
|---|---|---|---|---|
| dense_only    | 0.060 | 0.064 | 0.178 | 0.992 |
| bm25_only     | 0.230 | 0.261 | 0.448 | 0.980 |
| hybrid_rrf    | 0.170 | 0.184 | 0.409 | 0.989 |
| hybrid_rerank | 0.190 | 0.198 | 0.406 | 0.988 |

### Run 2 — labels additionally pooled from dense (bi-encoder) top-10

The dense retriever's top-10 per query were hand-judged and genuine relevant tracks
added to the label set.

| Config | P@10 | NDCG@10 | MRR | Novelty@10 |
|---|---|---|---|---|
| dense_only    | 0.580 | 0.619 | 0.750 | 0.992 |
| bm25_only     | 0.270 | 0.305 | 0.503 | 0.980 |
| hybrid_rrf    | 0.410 | 0.431 | 0.633 | 0.989 |
| hybrid_rerank | 0.340 | 0.345 | 0.561 | 0.988 |

---

## Labeling methodology and its effect on the numbers

Offline retrieval metrics score a system by membership in a fixed set of "relevant"
track ids. With an incomplete label set, any relevant-but-unlabeled track a system
retrieves is counted as **wrong** — so the *provenance* of the labels silently favors
whichever retrieval method they were drawn from.

- **Run 1** labels came from BM25 candidates, so they consist of tracks with high lexical
  overlap with the query. `dense_only` scored 0.064 NDCG — not because dense retrieval
  failed, but because it surfaced semantically-relevant tracks the BM25-sourced answer
  key did not contain. Root-cause inspection confirmed dense returned tight, on-query
  clusters (e.g. relaxing/melancholy jazz for the rainy-jazz query) with healthy cosine
  similarities (0.55–0.70); the model was correct and the labels were incomplete.

- **Run 2** added dense's own top-10 to the label set (pooling), and `dense_only` jumped
  to 0.619. `bm25_only` barely moved (0.261 → 0.305) — its hits were already labeled.
  This is the clean signature of the Run 1 bias.

- **Run 2 introduces the mirror bias.** Because dense's top-10 were pooled and judged,
  the label set now advantages `dense_only` by construction, while `hybrid_rrf` and
  `hybrid_rerank` draw candidates from a top-50 pool that was not fully judged. So the
  true config ranking lies **between** the two runs, not at either endpoint.

**The correct remedy** is a balanced pool: judge the top-k of *every* retrieval config
(dense, BM25, hybrid, reranked) before scoring, so no config is favored by label
provenance. That was scoped but not run here; it is the recommended next step and the ceiling on what these numbers can claim.

## What can and cannot be claimed

- **Defensible:** dense retrieval is *not* broken — it is the strongest single retriever
  on this catalog. The 4× deficit in Run 1 was a labeling artifact, root-caused and
  corrected.
- **Tentative:** cross-encoder reranking and RRF fusion do not beat pure dense here.
  This is partly real (ms-marco-MiniLM is trained on web search, not music semantics;
  BM25 is weak over homogeneous metadata-only descriptors) and partly the Run 2 label
  bias. Not cleanly separable without a balanced pool. **Do not** report "reranking hurts"
  as a clean result.

## Other known limitations

- **Homogeneous descriptors.** ~97% of the catalog is `metadata_only` (low lyrics
  coverage). Without lyrics the LLM descriptors collapse toward a template
  ("Mood: relaxing, meditative, sad. Tempo: moderate. Genre: Jazz…"), clustering
  embeddings tightly and limiting dense discrimination. Improving descriptor
  distinctiveness is the highest-leverage retrieval-quality lever.
- **Mixed index tiers in the aggregate.** With `index_tier="auto"`, 7/10 queries routed
  to `metadata_only` and 3/10 to `full_enrichment`. The aggregate NDCG mixes tiers;
  per-query tiers are recorded in `results.csv`. The dual-index "does lyrics grounding
  improve retrieval?" ablation is effectively untestable at this coverage (194 full-tier
  tracks) and is not claimed.
- **Novelty is uninformative here.** ~0.99 across all configs — max-normalized FMA play
  counts on a power-law distribution push nearly every track's popularity to ~0, so
  Novelty@10 does not discriminate between configs at this scale.

---

*Both runs reproduce from `python scripts/run_eval.py --config config/config.yaml --print-table`
against the corresponding `eval/queries.json` label set.*
