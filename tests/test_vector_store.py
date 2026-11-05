from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.processors import vector_store


def fake_emb_client(dim: int = 3) -> MagicMock:
    """Embeddings client that returns the items in reverse order, like a server may."""

    def create(input, model, encoding_format):
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text))] * dim)
            for i, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))

    client = MagicMock()
    client.embeddings.create.side_effect = create
    return client


def test_embed_dense_batches_and_keeps_input_order(conf):
    client = fake_emb_client()
    texts = ["a" * n for n in range(1, 6)]

    vectors = vector_store.embed_dense(client, conf, texts, batch_size=2)

    assert vectors == [[float(n)] * 3 for n in range(1, 6)]
    batches = [c.kwargs["input"] for c in client.embeddings.create.call_args_list]
    assert batches == [texts[0:2], texts[2:4], texts[4:5]]


def test_embed_dense_requests_float_encoding_with_configured_model(conf):
    client = fake_emb_client()
    vector_store.embed_dense(client, conf, ["text"])
    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == conf.embedder_name
    assert kwargs["encoding_format"] == "float"


def test_embed_dense_empty_input(conf):
    client = fake_emb_client()
    assert vector_store.embed_dense(client, conf, []) == []
    client.embeddings.create.assert_not_called()


def test_to_sparse_vector_converts_numpy_arrays():
    embedding = SimpleNamespace(
        indices=np.array([3, 7], dtype=np.int64), values=np.array([0.5, 1.5])
    )
    vector = vector_store.to_sparse_vector(embedding)
    assert vector.indices == [3, 7]
    assert vector.values == [0.5, 1.5]


def test_get_qdrant_client_defaults(monkeypatch, conf):
    qdrant_cls = MagicMock()
    monkeypatch.setattr(vector_store, "QdrantClient", qdrant_cls)
    vector_store.get_qdrant_client(conf)
    kwargs = qdrant_cls.call_args.kwargs
    assert kwargs["url"] == conf.qdrant_url
    assert kwargs["port"] == 443
    assert kwargs["https"] is True
    assert kwargs["verify"] is False
    assert kwargs["timeout"] == 30


def test_get_qdrant_client_optional_settings(monkeypatch, make_conf):
    qdrant_cls = MagicMock()
    monkeypatch.setattr(vector_store, "QdrantClient", qdrant_cls)
    vector_store.get_qdrant_client(make_conf(qdrant_verify_ssl=True, qdrant_timeout=5))
    kwargs = qdrant_cls.call_args.kwargs
    assert kwargs["verify"] is True
    assert kwargs["timeout"] == 5


def test_get_emb_client_uses_placeholder_key(conf):
    client = vector_store.get_emb_client(conf)
    assert client.api_key == "EMPTY"
    assert str(client.base_url).startswith(conf.emb_base_url)


def test_get_emb_client_uses_configured_key(make_conf):
    client = vector_store.get_emb_client(make_conf(emb_api_key="secret"))
    assert client.api_key == "secret"


def test_get_bm25_model_language(monkeypatch, make_conf):
    model_cls = MagicMock()
    monkeypatch.setattr(vector_store, "SparseTextEmbedding", model_cls)

    vector_store.get_bm25_model(make_conf())
    assert model_cls.call_args.kwargs["language"] == "english"

    vector_store.get_bm25_model(make_conf(bm25_language="russian"))
    assert model_cls.call_args.args == (vector_store.BM25_MODEL,)
    assert model_cls.call_args.kwargs == {
        "specific_model_path": vector_store.BM25_MODEL_PATH,
        "language": "russian",
    }
