# txt2vec-ds

Indexing of markdown documents into Qdrant and hybrid (dense + BM25) retrieval over them.

## Components

| File | Role |
|---|---|
| `src/processors/txt2vec.py` | `Txt2Vec`: splits a document into chunks, embeds them (dense + BM25) and upserts them into Qdrant. Also deletes documents by source. |
| `src/processors/data_getter.py` | `DataGetter`: hybrid search for a user query, optional reranking. |
| `src/processors/reranker.py` | `Reranker`: reorders candidates using the cross-encoder served by Infinity (`POST /rerank`). |
| `src/processors/generation_processor.py` | `Generation`: builds the prompt from the retrieved documents and calls the LLM (`src/llm_agent.py`). |
| `src/tracing.py` | Langfuse client setup and helpers for the query-path traces. |
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
4. Points are upserted in batches of 32; a batch rejected with 413 (proxy body size limit) is split in half until it fits. Then any points of the same `source` that weren't overwritten (left over from an older, longer version of the document) are deleted. Re-ingesting a source replaces it and doesn't create duplicates.

## Retrieval (`DataGetter`)

1. The query is embedded with the same dense model as at indexing time, and with BM25 `query_embed()`.
2. One `query_points` request runs two prefetches, `emb` and `bm25`, each returning `prefetch_k` candidates. They're merged with Reciprocal Rank Fusion (RRF) into the top `k_documents` candidates (100 by default).
3. If `tags` is non-empty, both prefetches are filtered with `tags` matching any of them. Empty tags or `None` means no filter.
4. If `use_reranker` is set, all `k_documents` candidates are reranked by Infinity (`reranker_base_url`, model `reranker_name`), and only the top `g_documents` (10 by default) are passed to the generator. Without the reranker, the top `g_documents` by RRF are passed. The reranker score is stored in `metadata["rerank_score"]`. If the reranker request fails, the error is logged and reported (`RERANKER_ERROR`), and the top `g_documents` by RRF are passed instead.

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
| `k_documents` | retrieval | candidates after fusion, input to the reranker (default `100`) |
| `g_documents` | retrieval | documents passed to the generator (default `10`), should be ≤ `k_documents` |
| `use_reranker` | retrieval | enable the reranker |

Optional (read with defaults):

| Field | Default | Meaning |
|---|---|---|
| `prefetch_k` | `50` | candidates per branch (dense / BM25) before fusion |
| `qdrant_timeout` | `30` | Qdrant request timeout, seconds |
| `emb_timeout` | `30` | embeddings request timeout, seconds |
| `reranker_timeout` | `30` | rerank request timeout, seconds |
| `qdrant_verify_ssl` | `False` | TLS certificate verification for Qdrant. Enable it if the certificate is valid. |
| `emb_verify_ssl` | `True` | TLS certificate verification for the Infinity embeddings server. `True`, `False`, or a path to a CA bundle. |
| `reranker_verify_ssl` | `True` | the same for the Infinity rerank server |
| `bm25_language` | `"english"` | BM25 stemmer and stopwords language, e.g. `"russian"` |
| `langfuse_host` | `''` | Langfuse server URL |
| `langfuse_public_key`, `langfuse_secret_key` | `''` | Langfuse project keys. Tracing is off while either is empty. Set them through the environment. |
| `langfuse_environment` | `''` | Langfuse environment label, e.g. `test`, `prod` |

The BM25 model files are loaded from the local `models` directory.

The embedder and the reranker can run on one Infinity server (point both URLs at it) or on separate servers. `embedder_name` and `reranker_name` must match the model ids that Infinity serves (`--served-model-name`, or `--model-id` if that isn't set). Include Infinity's `--url-prefix` in the base URLs if one is set.

## Observability (Langfuse)

Each user message becomes one Langfuse trace (SDK v3, needs Langfuse server v3+):

```
trace "rag-query"   session = conversation_id, tags = query tags, input = message, output = response
├─ retrieval          DataGetter: the documents passed to the generator (source, headers, scores, text)
│  ├─ hybrid-search   query, tags, limits; candidates with RRF scores
│  └─ rerank          index -> relevance_score; level ERROR if the reranker failed
└─ generation         Generation: response, doc_array
   └─ llm             the LLM call (langfuse.openai): messages, model, token usage, latency
```

`DataGetter` starts the trace and stores its id in `MessageEvent.trace_id`, and `Generation` adds its spans to the same trace. `MessageEvent` needs a `trace_id: str | None = None` field for this. Without it, a warning is logged, and the generation goes into a separate trace.

Tracing is disabled while `langfuse_public_key` or `langfuse_secret_key` is empty. Spans are exported in the background, and export failures don't affect requests. Indexing (`Txt2Vec`, `scripts/load_docs.py`) isn't traced.

If the Langfuse server uses a certificate from an internal CA, point `OTEL_EXPORTER_OTLP_CERTIFICATE` (span export) and `SSL_CERT_FILE` (API calls) at the CA bundle.

## Checking parsed documents

The PDF parser sometimes drops parts of a document, which shows up as a jump in the numbering (e.g. 7.6 is followed by 7.9). Download the files (`scripts/download_s3.py`) and check them before indexing:

```
python scripts/check_numbering.py s3_dump --report numbering_report.csv
```

Numbered lines are found at the line start: plain (`7.6.`, `3)`), in markdown headers (`## 7.6 Title`) or after a keyword (`Статья 7.6`, `Глава 3`, `п. 2.1`). Articles, chapters, sections and points are checked as separate sequences. Nested lists and restarts at 1 (e.g. the body after the table of contents) aren't reported.

The CSV report (`file, stream, line, prev, got, missing, kind, text`) has two kinds of rows:

- `gap`: numbers are missing, most likely lost text. Re-parse the document.
- `out_of_order`: a number that doesn't continue the sequence (a cross-reference at the line start, a stray number, or a jump bigger than `--max-gap`, 20 by default). Worth a look, but usually not lost text.

The script exits with 1 if any gap is found. Files without numbered lines are listed in the log as not checked.

## Re-indexing

Re-index the whole collection (drop it and ingest all documents again) after changing any of these:

- `embedder_name` or `emb_size`
- `bm25_language`
- the chunking, or the text that is embedded (header path)

Queries have to be embedded exactly the way the documents were.
