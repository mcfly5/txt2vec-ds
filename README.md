# txt2vec-ds

Indexing of markdown documents into Qdrant and hybrid (dense + BM25) retrieval over them.

## Components

| File | Role |
|---|---|
| `src/processors/txt2vec.py` | `Txt2Vec`: splits a document into chunks, embeds them (dense + BM25) and upserts them into Qdrant. Also deletes documents by source. |
| `src/processors/data_getter.py` | `DataGetter`: hybrid search for a user query, optional reranking. |
| `src/processors/reranker.py` | `Reranker`: reorders candidates using the cross-encoder served by Infinity (`POST /rerank`). |
| `src/processors/vector_store.py` | Settings shared by indexing and retrieval: vector names, BM25 model, Qdrant and embedding clients. Both sides must use the same settings, so change them only here. |

## Collection schema

Each point is one chunk:

- vectors
  - `emb`: dense, cosine distance, size `emb_size`, produced by Infinity's OpenAI-compatible embeddings API (`emb_base_url`, model `embedder_name`)
  - `bm25`: sparse, `Modifier.IDF`, produced by fastembed `Qdrant/BM25` (IDF is applied by Qdrant at search time)
- payload
  - `document`: chunk text
  - `metadata`: `source`, `Header 1..3`, `part#`, `child#` (only for split sections), `tokens`
  - `tags`: document tags (top level, used for filtering)
- payload indexes (keyword): `tags`, `metadata.source`
- point id: `uuid5(source + chunk index)`, so it's deterministic

The collection and payload indexes are created by `Txt2Vec` on startup if they don't exist.

## Indexing (`Txt2Vec`)

1. The markdown is split by `#`, `##` and `###` headers. Sections longer than 500 tokens (`tokenizer_name`) are split again, with no overlap.
2. The header path (`Header 1 > Header 2 > ...`) is prepended to each chunk's text before embedding. The splitter strips headers from the content, and adding them back gives both vectors the section context. The payload `document` stores the chunk text without the header path.
3. Dense embeddings are requested in batches of 32. BM25 embeddings use `embed()`.
4. Points are upserted in batches of 128. Then any points of the same `source` that weren't overwritten (left over from an older, longer version of the document) are deleted. Re-ingesting a source replaces it and doesn't create duplicates.

## Retrieval (`DataGetter`)

1. The query is embedded with the same dense model as at indexing time, and with BM25 `query_embed()`.
2. One `query_points` request runs two prefetches, `emb` and `bm25`, each returning `prefetch_k` candidates. They're merged with Reciprocal Rank Fusion (RRF) into the top `k_document`.
3. If `tags` is non-empty, both prefetches are filtered with `tags` matching any of them. Empty tags or `None` means no filter.
4. If `use_reranker` is set, the candidates are reranked by Infinity (`reranker_base_url`, model `reranker_name`), and the top `r_documents` are returned. The reranker score is stored in `metadata["rerank_score"]`. If the reranker request fails, the error is logged and reported (`RERANKER_ERROR`), and the documents keep their RRF order.

Results are `langchain_core.documents.Document` objects. `metadata` contains the payload metadata plus `tags`, `id` and `score`. The score is the RRF score, which is rank-based and not comparable to cosine similarity, so don't threshold it.

## Configuration

Used from `Configuration`:

| Field | Used by | Meaning |
|---|---|---|
| `qdrant_url`, `qdrant_collection` | both | Qdrant endpoint (HTTPS, port 443) and collection |
| `emb_base_url`, `emb_api_key`, `embedder_name` | both | Infinity embeddings endpoint, API key (empty if Infinity has no `--api-key`) and model |
| `reranker_base_url`, `reranker_api_key`, `reranker_name` | retrieval | Infinity rerank endpoint, API key and model |
| `emb_size` | indexing | dense vector size (used when creating the collection) |
| `tokenizer_name` | indexing | HF tokenizer used for chunk sizing |
| `k_document` | retrieval | candidates after fusion (input to the reranker) |
| `r_documents` | retrieval | documents returned |
| `use_reranker` | retrieval | enable the reranker |

Optional (read with defaults):

| Field | Default | Meaning |
|---|---|---|
| `prefetch_k` | `50` | candidates per branch (dense / BM25) before fusion |
| `qdrant_timeout` | `30` | Qdrant request timeout, seconds |
| `emb_timeout` | `30` | embeddings request timeout, seconds |
| `reranker_timeout` | `30` | rerank request timeout, seconds |
| `qdrant_verify_ssl` | `False` | TLS certificate verification for Qdrant. Enable it if the certificate is valid. |
| `bm25_language` | `"english"` | BM25 stemmer and stopwords language, e.g. `"russian"` |

The BM25 model files are loaded from the local `models` directory.

The embedder and the reranker can run on one Infinity server (point both URLs at it) or on separate servers. `embedder_name` and `reranker_name` must match the model ids that Infinity serves (`--served-model-name`, or `--model-id` if that isn't set). Include Infinity's `--url-prefix` in the base URLs if one is set.

## Re-indexing

Re-index the whole collection (drop it and ingest all documents again) after changing any of these:

- `embedder_name` or `emb_size`
- `bm25_language`
- the chunking, or the text that is embedded (header path)

Queries have to be embedded exactly the way the documents were.
