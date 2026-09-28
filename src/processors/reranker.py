from typing import List

import httpx
from blocks import Processor
from langchain_core.documents import Document
from loguru import logger

from config import Configuration
from src.utils.utils import Measures, ErrorCodes, metric_sender


class Reranker(Processor):
    """Reranks documents with the cross-encoder served by Infinity (POST /rerank)."""

    def __init__(self, conf: Configuration) -> None:
        self.conf = conf
        headers = (
            {"Authorization": f"Bearer {conf.reranker_api_key}"}
            if conf.reranker_api_key
            else None
        )
        self.client = httpx.Client(
            base_url=conf.reranker_base_url,
            headers=headers,
            timeout=getattr(conf, "reranker_timeout", 30),
            verify=getattr(conf, "reranker_verify_ssl", True),
        )
        logger.info(f'Reranker: {conf.reranker_name} at {conf.reranker_base_url}')

    def __call__(
        self, query: str, documents: List[Document], top_n: int | None = None
    ) -> List[Document]:
        logger.info(f'calling Reranker with query: {query}')
        return self.rerank(query, documents, top_n)

    def rerank(
        self, query: str, documents: List[Document], top_n: int | None = None
    ) -> List[Document]:
        """Return the `top_n` most relevant documents (all of them if `top_n` is None)."""
        payload = {
            'model': self.conf.reranker_name,
            'query': query,
            'documents': [doc.page_content for doc in documents],
            'return_documents': False,
        }
        if top_n is not None:
            payload['top_n'] = top_n
        try:
            response = self.client.post('/rerank', json=payload)
            response.raise_for_status()
            results = response.json()['results']
        except Exception as err:
            # Reranking only improves the order; keep the fused order if it's unavailable.
            logger.error(f"Reranker request failed, keeping retrieval order: {err}")
            metric_sender.send_metric(
                name=Measures.error_code,
                value=ErrorCodes.RERANKER_ERROR)
            return documents[:top_n]

        results = sorted(results, key=lambda r: r['relevance_score'], reverse=True)[:top_n]
        logger.debug(f'Reranking scores: {[r["relevance_score"] for r in results]}')
        reranked = []
        for result in results:
            doc = documents[result['index']]
            doc.metadata['rerank_score'] = result['relevance_score']
            reranked.append(doc)
        logger.info('Reranked documents:')
        logger.info([doc.metadata for doc in reranked])

        return reranked
