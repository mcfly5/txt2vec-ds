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
        )
        logger.info(f'Reranker: {conf.reranker_name} at {conf.reranker_base_url}')

    def __call__(self, query: str, documents: List[Document]) -> List[Document]:
        logger.info(f'calling Reranker with query: {query}')
        return self.rerank(query, documents)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        try:
            response = self.client.post(
                '/rerank',
                json={
                    'model': self.conf.reranker_name,
                    'query': query,
                    'documents': [doc.page_content for doc in documents],
                    'return_documents': False,
                },
            )
            response.raise_for_status()
            results = response.json()['results']
        except Exception as err:
            # Reranking only improves the order; keep the fused order if it's unavailable.
            logger.error(f"Reranker request failed, keeping retrieval order: {err}")
            metric_sender.send_metric(
                name=Measures.error_code,
                value=ErrorCodes.RERANKER_ERROR)
            return documents

        results = sorted(results, key=lambda r: r['relevance_score'], reverse=True)
        logger.debug(f'Reranking scores: {[r["relevance_score"] for r in results]}')
        reranked = []
        for result in results:
            doc = documents[result['index']]
            doc.metadata['rerank_score'] = result['relevance_score']
            reranked.append(doc)
        logger.info('Reranked documents:')
        logger.info([doc.metadata for doc in reranked])

        return reranked
