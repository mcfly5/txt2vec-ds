import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from langchain_core.documents import Document
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointStruct,
    models,
)

from src.events import InputEvent, PreprocessedData
from src.processors import txt2vec
from src.processors.txt2vec import UPSERT_BATCH_SIZE, Txt2Vec
from src.processors.vector_store import DENSE_VECTOR, SPARSE_VECTOR

DOC = """# Intro
Intro text.

## Scope
Scope text.

### Details
Details text.

# Second
Second text.
"""


def sparse(i: int) -> SimpleNamespace:
    return SimpleNamespace(indices=np.array([i]), values=np.array([1.0]))


@pytest.fixture
def qdrant():
    client = MagicMock(name="qdrant")
    client.collection_exists.return_value = True
    return client


@pytest.fixture
def bm25():
    model = MagicMock(name="bm25")
    model.embed.side_effect = lambda texts: (sparse(i) for i, _ in enumerate(texts))
    return model


@pytest.fixture
def embed_dense(monkeypatch):
    embed = MagicMock(
        name="embed_dense",
        side_effect=lambda client, conf, texts: [[float(i)] * 4 for i in range(len(texts))],
    )
    monkeypatch.setattr(txt2vec, "embed_dense", embed)
    return embed


@pytest.fixture
def make_txt2vec(monkeypatch, qdrant, bm25, embed_dense, word_tokenizer, make_conf):
    monkeypatch.setattr(txt2vec, "get_qdrant_client", lambda conf: qdrant)
    monkeypatch.setattr(txt2vec, "get_emb_client", MagicMock(name="emb_client"))
    monkeypatch.setattr(txt2vec, "get_bm25_model", lambda conf: bm25)
    monkeypatch.setattr(
        txt2vec.AutoTokenizer, "from_pretrained", lambda name: word_tokenizer
    )
    return lambda **conf: Txt2Vec(make_conf(**conf))


def upserted_points(qdrant) -> list[PointStruct]:
    return [p for c in qdrant.upsert.call_args_list for p in c.kwargs["points"]]


def too_large() -> UnexpectedResponse:
    return UnexpectedResponse(413, "Request Entity Too Large", b"", httpx.Headers())


# --- collection ---------------------------------------------------------------


def test_creates_missing_collection(make_txt2vec, qdrant):
    qdrant.collection_exists.return_value = False

    make_txt2vec(emb_size=1024)

    kwargs = qdrant.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "documents"
    dense = kwargs["vectors_config"][DENSE_VECTOR]
    assert (dense.size, dense.distance) == (1024, Distance.COSINE)
    assert kwargs["sparse_vectors_config"][SPARSE_VECTOR].modifier == models.Modifier.IDF


def test_keeps_existing_collection(make_txt2vec, qdrant):
    make_txt2vec()
    qdrant.create_collection.assert_not_called()


def test_creates_payload_indexes(make_txt2vec, qdrant):
    make_txt2vec()
    indexes = {
        c.kwargs["field_name"]: c.kwargs["field_schema"]
        for c in qdrant.create_payload_index.call_args_list
    }
    assert indexes == {
        "tags": models.PayloadSchemaType.KEYWORD,
        "metadata.source": models.PayloadSchemaType.KEYWORD,
    }


def test_collection_error_is_reported(make_txt2vec, qdrant, sent_errors):
    qdrant.collection_exists.side_effect = RuntimeError("unreachable")
    with pytest.raises(RuntimeError):
        make_txt2vec()
    assert sent_errors() == ["QDRANT_ERROR"]


# --- chunking -----------------------------------------------------------------


def test_split_doc_by_headers(make_txt2vec, word_tokenizer):
    chunks = make_txt2vec().split_doc(DOC, word_tokenizer)

    assert [c.page_content.strip() for c in chunks] == [
        "Intro text.",
        "Scope text.",
        "Details text.",
        "Second text.",
    ]
    assert chunks[2].metadata["Header 1"] == "Intro"
    assert chunks[2].metadata["Header 2"] == "Scope"
    assert chunks[2].metadata["Header 3"] == "Details"
    assert chunks[3].metadata["Header 1"] == "Second"
    assert "Header 2" not in chunks[3].metadata
    assert [c.metadata["part#"] for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].metadata["tokens"] == 3  # "Intro", "text", "."
    assert all("child#" not in c.metadata for c in chunks)


def test_split_doc_splits_long_sections(make_txt2vec, word_tokenizer):
    paragraphs = "\n\n".join(" ".join(["word"] * 100) for _ in range(12))
    doc = f"# Long\n{paragraphs}\n\n# Short\nshort text"

    chunks = make_txt2vec().split_doc(doc, word_tokenizer)

    long_chunks = [c for c in chunks if c.metadata["Header 1"] == "Long"]
    assert len(long_chunks) > 1
    assert [c.metadata["child#"] for c in long_chunks] == list(range(len(long_chunks)))
    assert all(c.metadata["part#"] == 0 for c in long_chunks)
    assert all(c.metadata["tokens"] <= 500 for c in long_chunks)
    assert sum(c.page_content.count("word") for c in long_chunks) == 1200
    assert chunks[-1].metadata["part#"] == 1
    assert "child#" not in chunks[-1].metadata


def test_text_to_embed_prepends_header_path():
    chunk = Document(
        page_content="body", metadata={"Header 1": "A", "Header 2": "B", "part#": 0}
    )
    assert Txt2Vec._text_to_embed(chunk) == "A > B\n\nbody"


def test_text_to_embed_without_headers():
    assert Txt2Vec._text_to_embed(Document(page_content="body")) == "body"


# --- upsert -------------------------------------------------------------------


def test_upsert_document_points(make_txt2vec, qdrant, embed_dense, bm25):
    make_txt2vec()._upsert_document(DOC, "s3://bucket/a.md", ["kb"])

    points = upserted_points(qdrant)
    assert len(points) == 4
    assert [p.id for p in points] == [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"s3://bucket/a.md:{i}")) for i in range(4)
    ]
    first = points[2]
    assert first.payload["document"].strip() == "Details text."
    assert first.payload["tags"] == ["kb"]
    assert first.payload["metadata"]["source"] == "s3://bucket/a.md"
    assert first.payload["metadata"]["Header 3"] == "Details"
    assert first.vector[DENSE_VECTOR] == [2.0] * 4
    assert first.vector[SPARSE_VECTOR] == models.SparseVector(indices=[2], values=[1.0])

    # Both embeddings see the header path, the payload doesn't.
    texts = embed_dense.call_args.args[2]
    assert texts[2].startswith("Intro > Scope > Details\n\n")
    assert bm25.embed.call_args.args[0] == texts


def test_upsert_is_deterministic(make_txt2vec, qdrant):
    t2v = make_txt2vec()
    t2v._upsert_document(DOC, "a.md", [])
    t2v._upsert_document(DOC, "a.md", [])
    ids = [p.id for p in upserted_points(qdrant)]
    assert ids[:4] == ids[4:]


def test_upsert_deletes_stale_points_of_source(make_txt2vec, qdrant):
    make_txt2vec()._upsert_document(DOC, "a.md", [])

    ids = [p.id for p in upserted_points(qdrant)]
    assert qdrant.delete.call_args.kwargs["points_selector"] == Filter(
        must=[FieldCondition(key="metadata.source", match=MatchValue(value="a.md"))],
        must_not=[HasIdCondition(has_id=ids)],
    )


def test_upsert_in_batches(make_txt2vec, qdrant):
    n = UPSERT_BATCH_SIZE * 2 + 5
    doc = "\n\n".join(f"# H{i}\ntext {i}" for i in range(n))

    make_txt2vec()._upsert_document(doc, "a.md", [])

    sizes = [len(c.kwargs["points"]) for c in qdrant.upsert.call_args_list]
    assert sizes == [UPSERT_BATCH_SIZE, UPSERT_BATCH_SIZE, 5]


def test_upsert_empty_document_is_skipped(make_txt2vec, qdrant, embed_dense):
    make_txt2vec()._upsert_document("", "a.md", [])
    embed_dense.assert_not_called()
    qdrant.upsert.assert_not_called()
    qdrant.delete.assert_not_called()


def test_upsert_embedding_error(make_txt2vec, qdrant, embed_dense, sent_errors):
    embed_dense.side_effect = RuntimeError("infinity is down")
    with pytest.raises(RuntimeError):
        make_txt2vec()._upsert_document(DOC, "a.md", [])
    assert sent_errors() == ["EMBEDDINGS_ERROR"]
    qdrant.upsert.assert_not_called()


def test_upsert_embedding_count_mismatch(make_txt2vec, qdrant, embed_dense):
    embed_dense.side_effect = lambda client, conf, texts: [[0.0] * 4]
    with pytest.raises(ValueError, match="count mismatch"):
        make_txt2vec()._upsert_document(DOC, "a.md", [])
    qdrant.upsert.assert_not_called()


def test_upsert_qdrant_error_keeps_old_points(make_txt2vec, qdrant, sent_errors):
    qdrant.upsert.side_effect = RuntimeError("timeout")
    with pytest.raises(RuntimeError):
        make_txt2vec()._upsert_document(DOC, "a.md", [])
    assert sent_errors() == ["QDRANT_ERROR"]
    qdrant.delete.assert_not_called()


# --- 413 handling ---------------------------------------------------------------


def make_points(n: int) -> list[PointStruct]:
    return [PointStruct(id=i, vector=[0.0], payload={}) for i in range(n)]


def test_upsert_batch_splits_on_413(make_txt2vec, qdrant):
    # The proxy accepts at most 2 points per request.
    def upsert(collection_name, points):
        if len(points) > 2:
            raise too_large()

    qdrant.upsert.side_effect = upsert

    make_txt2vec()._upsert_batch(make_points(7))

    accepted = [
        c.kwargs["points"]
        for c in qdrant.upsert.call_args_list
        if len(c.kwargs["points"]) <= 2
    ]
    assert [p.id for batch in accepted for p in batch] == list(range(7))


def test_upsert_batch_single_point_413_is_raised(make_txt2vec, qdrant):
    qdrant.upsert.side_effect = too_large()
    with pytest.raises(UnexpectedResponse):
        make_txt2vec()._upsert_batch(make_points(4))
    # 4 -> 2 -> 1, then gives up
    assert [len(c.kwargs["points"]) for c in qdrant.upsert.call_args_list] == [4, 2, 1]


def test_upsert_batch_other_errors_are_not_split(make_txt2vec, qdrant):
    qdrant.upsert.side_effect = UnexpectedResponse(500, "error", b"", httpx.Headers())
    with pytest.raises(UnexpectedResponse):
        make_txt2vec()._upsert_batch(make_points(4))
    qdrant.upsert.assert_called_once()


# --- processor --------------------------------------------------------------------


def test_call_completed(make_txt2vec, qdrant):
    event = InputEvent(tags=["kb"])
    status = make_txt2vec()(PreprocessedData(data=DOC, source="a.md", event=event))

    assert status.status == "completed"
    assert status.event is event
    assert upserted_points(qdrant)[0].payload["tags"] == ["kb"]


def test_call_failed(make_txt2vec, qdrant):
    qdrant.upsert.side_effect = RuntimeError("timeout")
    status = make_txt2vec()(PreprocessedData(data=DOC, source="a.md", event=InputEvent()))
    assert status.status == "failed"


# --- delete -------------------------------------------------------------------------


def test_delete_documents_by_source(make_txt2vec, qdrant):
    qdrant.count.return_value = SimpleNamespace(count=3)

    result = make_txt2vec().delete_documents_by_source("a.md")

    source_filter = Filter(
        must=[FieldCondition(key="metadata.source", match=MatchValue(value="a.md"))]
    )
    assert qdrant.count.call_args.kwargs["count_filter"] == source_filter
    assert qdrant.delete.call_args.kwargs["points_selector"] == source_filter
    assert result["status"] == "success"
    assert result["deleted_count"] == 3


def test_delete_documents_by_source_nothing_found(make_txt2vec, qdrant):
    qdrant.count.return_value = SimpleNamespace(count=0)

    result = make_txt2vec().delete_documents_by_source("a.md")

    qdrant.delete.assert_not_called()
    assert result["deleted_count"] == 0


def test_delete_documents_by_source_error(make_txt2vec, qdrant):
    qdrant.count.side_effect = RuntimeError("timeout")
    result = make_txt2vec().delete_documents_by_source("a.md")
    assert result["status"] == "error"
    assert "timeout" in result["message"]
