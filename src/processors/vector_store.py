"""Shared Qdrant / embedding setup for ingestion (Txt2Vec) and retrieval (DataGetter).

Everything that must be identical on both sides (vector names, BM25 model and
language, dense embedding model) lives here so indexing and querying cannot drift.
"""

from config import Configuration
from fastembed import SparseTextEmbedding
from openai import Client
from qdrant_client import QdrantClient, models

DENSE_VECTOR = "emb"
SPARSE_VECTOR = "bm25"
BM25_MODEL = "Qdrant/BM25"
BM25_MODEL_PATH = "models"


def get_qdrant_client(conf: Configuration) -> QdrantClient:
    return QdrantClient(
        url=conf.qdrant_url,
        port=443,
        https=True,
        verify=getattr(conf, "qdrant_verify_ssl", False),
        timeout=getattr(conf, "qdrant_timeout", 30),
    )


def get_emb_client(conf: Configuration) -> Client:
    return Client(base_url=conf.llm_base_url, api_key=conf.llm_api_key)


def get_bm25_model(conf: Configuration) -> SparseTextEmbedding:
    # Stemmer/stopwords language must be the same for indexing and querying;
    # changing it requires re-indexing the collection.
    return SparseTextEmbedding(
        BM25_MODEL,
        specific_model_path=BM25_MODEL_PATH,
        language=getattr(conf, "bm25_language", "english"),
    )


def embed_dense(
    client: Client, conf: Configuration, texts: list[str], batch_size: int = 32
) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), batch_size):
        data = client.embeddings.create(
            input=texts[start : start + batch_size], model=conf.embedder_name
        ).data
        vectors.extend(item.embedding for item in sorted(data, key=lambda d: d.index))
    return vectors


def to_sparse_vector(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )
