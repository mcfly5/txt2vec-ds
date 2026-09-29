from blocks import Processor
from config import Configuration
from langchain_core.documents import Document
from langfuse import get_client, observe
from loguru import logger
from qdrant_client import models
from src.events import Documents, MessageEvent
from src.processors.reranker import Reranker
from src.processors.vector_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    embed_dense,
    get_bm25_model,
    get_emb_client,
    get_qdrant_client,
    to_sparse_vector,
)
from src.tracing import doc_summary, init_langfuse, new_trace_id, set_trace_id
from src.utils.utils import ErrorCodes, Measures, metric_sender


class DataGetter(Processor):
    def __init__(self, conf: Configuration) -> None:
        self.conf = conf
        self.langfuse = init_langfuse(conf)
        try:
            self.emb_client = get_emb_client(conf)
            self.bm25_model = get_bm25_model(conf)
        except Exception as err:
            logger.error("Can't load embedding models")
            metric_sender.send_metric(
                name=Measures.error_code, value=ErrorCodes.EMBEDDINGS_ERROR
            )
            raise err
        self.client = get_qdrant_client(conf)
        self.reranker = Reranker(conf) if conf.use_reranker else None

    def __call__(self, query: MessageEvent) -> Documents:
        logger.info(f"calling DataGetter with tags: {query.tags}")
        trace_id = new_trace_id()
        set_trace_id(query, trace_id)
        with self.langfuse.start_as_current_span(
            name="retrieval", trace_context={"trace_id": trace_id}
        ) as span:
            span.update_trace(
                name="rag-query",
                session_id=(
                    str(query.conversation_id) if query.conversation_id else None
                ),
                tags=query.tags or [],
                input=query.message,
            )
            docs = self.retrieve(query.message, query.tags, self.conf.k_documents)
            if self.reranker and docs:
                logger.info("Using reranker")
                docs = self.reranker(query.message, docs, top_n=self.conf.g_documents)
            # Also covers use_reranker=False: the generator gets the top RRF results.
            docs = docs[: self.conf.g_documents]
            logger.info(f"{len(docs)} docs passed to generator")
            span.update(output=[doc_summary(doc, with_text=True) for doc in docs])
        return Documents(query=query.message, documents=docs, event=query)

    @observe(name="hybrid-search", capture_input=False, capture_output=False)
    def retrieve(
        self, query: str, tags: list[str] | None, k_documents: int = 10
    ) -> list[Document]:
        """Hybrid search: dense + BM25 candidates fused with RRF."""
        langfuse = get_client()
        try:
            dense_vector = embed_dense(self.emb_client, self.conf, [query])[0]
            sparse_vector = to_sparse_vector(
                next(iter(self.bm25_model.query_embed(query)))
            )
        except Exception as err:
            logger.error(f"Can't embed query: {err}")
            langfuse.update_current_span(level="ERROR", status_message=f"embed: {err}")
            metric_sender.send_metric(
                name=Measures.error_code, value=ErrorCodes.EMBEDDINGS_ERROR
            )
            raise err

        tags_filter = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="tags", match=models.MatchAny(any=tags)
                    )
                ]
            )
            if tags
            else None
        )
        prefetch_limit = max(getattr(self.conf, "prefetch_k", 50), k_documents)
        langfuse.update_current_span(
            input={
                "query": query,
                "tags": tags,
                "k_documents": k_documents,
                "prefetch_limit": prefetch_limit,
            }
        )

        try:
            points = self.client.query_points(
                collection_name=self.conf.qdrant_collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR,
                        filter=tags_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR,
                        filter=tags_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k_documents,
                with_payload=True,
            ).points
        except Exception as err:
            logger.error(f"Qdrant query failed: {err}")
            langfuse.update_current_span(level="ERROR", status_message=f"qdrant: {err}")
            metric_sender.send_metric(
                name=Measures.error_code, value=ErrorCodes.QDRANT_ERROR
            )
            raise err

        docs = [
            Document(
                page_content=point.payload["document"],
                metadata={
                    **point.payload.get("metadata", {}),
                    "tags": point.payload.get("tags", []),
                    "id": point.id,
                    "score": point.score,
                },
            )
            for point in points
        ]
        logger.info(f"{len(docs)} docs retrieved")
        langfuse.update_current_span(output=[doc_summary(doc) for doc in docs])
        return docs
