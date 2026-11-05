from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document
from qdrant_client import models

from src.events import MessageEvent, ResponseErrorCode, ResponseStatus
from src.processors import data_getter
from src.processors.data_getter import DataGetter, RetrievalError
from src.processors.vector_store import DENSE_VECTOR, SPARSE_VECTOR


def point(i: int, score: float = 0.5, **metadata) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"id-{i}",
        score=score,
        payload={
            "document": f"chunk {i}",
            "metadata": {"source": f"doc{i}.md", **metadata},
            "tags": ["kb"],
        },
    )


@pytest.fixture
def qdrant():
    client = MagicMock(name="qdrant")
    client.query_points.return_value = SimpleNamespace(points=[point(0), point(1)])
    return client


@pytest.fixture
def bm25():
    model = MagicMock(name="bm25")
    model.query_embed.side_effect = lambda q: iter(
        [SimpleNamespace(indices=np.array([1, 2]), values=np.array([0.3, 0.7]))]
    )
    return model


@pytest.fixture
def embed_dense(monkeypatch):
    embed = MagicMock(name="embed_dense", return_value=[[0.1, 0.2, 0.3, 0.4]])
    monkeypatch.setattr(data_getter, "embed_dense", embed)
    return embed


@pytest.fixture
def reranker_cls(monkeypatch):
    cls = MagicMock(name="Reranker")
    # By default the reranker reverses the order and returns top_n.
    cls.return_value.side_effect = lambda q, docs, top_n=None: list(reversed(docs))[:top_n]
    monkeypatch.setattr(data_getter, "Reranker", cls)
    return cls


@pytest.fixture
def make_getter(monkeypatch, qdrant, bm25, embed_dense, reranker_cls, make_conf):
    monkeypatch.setattr(data_getter, "get_emb_client", MagicMock(name="emb_client"))
    monkeypatch.setattr(data_getter, "get_bm25_model", lambda conf: bm25)
    monkeypatch.setattr(data_getter, "get_qdrant_client", lambda conf: qdrant)
    return lambda **conf: DataGetter(make_conf(**conf))


def make_event(tags=None) -> MessageEvent:
    return MessageEvent(message="what is x?", conversation_id=7, tags=tags or [])


def query_kwargs(qdrant) -> dict:
    return qdrant.query_points.call_args.kwargs


# --- retrieve -----------------------------------------------------------------


def test_retrieve_builds_documents_from_points(make_getter, qdrant):
    qdrant.query_points.return_value = SimpleNamespace(
        points=[point(3, score=0.8, **{"Header 1": "Intro", "part#": 0})]
    )
    getter = make_getter()

    docs = getter.retrieve("query", tags=None, k_documents=5)

    assert docs == [
        Document(
            page_content="chunk 3",
            metadata={
                "source": "doc3.md",
                "Header 1": "Intro",
                "part#": 0,
                "tags": ["kb"],
                "id": "id-3",
                "score": 0.8,
            },
        )
    ]


def test_retrieve_runs_hybrid_query_with_rrf(make_getter, qdrant, embed_dense):
    getter = make_getter()
    getter.retrieve("query", tags=None, k_documents=5)

    kwargs = query_kwargs(qdrant)
    assert kwargs["collection_name"] == "documents"
    assert kwargs["limit"] == 5
    assert kwargs["with_payload"] is True
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)

    dense, sparse = kwargs["prefetch"]
    assert dense.using == DENSE_VECTOR
    assert dense.query == [0.1, 0.2, 0.3, 0.4]
    assert sparse.using == SPARSE_VECTOR
    assert sparse.query == models.SparseVector(indices=[1, 2], values=[0.3, 0.7])
    assert embed_dense.call_args.args[2] == ["query"]


@pytest.mark.parametrize("tags", [None, []])
def test_retrieve_without_tags_has_no_filter(make_getter, qdrant, tags):
    make_getter().retrieve("query", tags=tags, k_documents=5)
    assert all(p.filter is None for p in query_kwargs(qdrant)["prefetch"])


def test_retrieve_filters_both_branches_by_any_tag(make_getter, qdrant):
    make_getter().retrieve("query", tags=["a", "b"], k_documents=5)

    expected = models.Filter(
        must=[models.FieldCondition(key="tags", match=models.MatchAny(any=["a", "b"]))]
    )
    assert [p.filter for p in query_kwargs(qdrant)["prefetch"]] == [expected, expected]


@pytest.mark.parametrize(
    ("conf", "k_documents", "expected"),
    [
        ({}, 10, 50),  # default prefetch_k
        ({}, 100, 100),  # never less than k_documents
        ({"prefetch_k": 20}, 10, 20),
    ],
)
def test_retrieve_prefetch_limit(make_getter, qdrant, conf, k_documents, expected):
    make_getter(**conf).retrieve("query", tags=None, k_documents=k_documents)
    assert [p.limit for p in query_kwargs(qdrant)["prefetch"]] == [expected, expected]


def test_retrieve_embedding_error(make_getter, embed_dense, qdrant, sent_errors):
    embed_dense.side_effect = RuntimeError("infinity is down")

    with pytest.raises(RetrievalError) as exc:
        make_getter().retrieve("query", tags=None)

    assert exc.value.code == ResponseErrorCode.EMBEDDINGS_ERROR
    assert "infinity is down" in str(exc.value)
    assert sent_errors() == ["EMBEDDINGS_ERROR"]
    qdrant.query_points.assert_not_called()


def test_retrieve_bm25_error_is_embedding_error(make_getter, bm25):
    bm25.query_embed.side_effect = RuntimeError("bm25")
    with pytest.raises(RetrievalError) as exc:
        make_getter().retrieve("query", tags=None)
    assert exc.value.code == ResponseErrorCode.EMBEDDINGS_ERROR


def test_retrieve_qdrant_error(make_getter, qdrant, sent_errors):
    qdrant.query_points.side_effect = RuntimeError("timeout")

    with pytest.raises(RetrievalError) as exc:
        make_getter().retrieve("query", tags=None)

    assert exc.value.code == ResponseErrorCode.QDRANT_ERROR
    assert sent_errors() == ["QDRANT_ERROR"]


# --- __call__ -----------------------------------------------------------------


def test_call_returns_documents_and_sets_trace_id(make_getter):
    event = make_event(tags=["kb"])

    result = make_getter()(event)

    assert result.query == "what is x?"
    assert result.event is event
    assert [d.page_content for d in result.documents] == ["chunk 0", "chunk 1"]
    assert event.status == ResponseStatus.COMPLETED
    assert event.trace_id and len(event.trace_id) == 32


def test_call_passes_k_documents_and_truncates_to_g_documents(make_getter, qdrant):
    qdrant.query_points.return_value = SimpleNamespace(points=[point(i) for i in range(5)])

    result = make_getter(k_documents=5, g_documents=2)(make_event())

    assert query_kwargs(qdrant)["limit"] == 5
    assert [d.page_content for d in result.documents] == ["chunk 0", "chunk 1"]


def test_call_reranks_with_top_n_g_documents(make_getter, qdrant, reranker_cls):
    qdrant.query_points.return_value = SimpleNamespace(points=[point(i) for i in range(5)])

    result = make_getter(use_reranker=True, k_documents=5, g_documents=2)(make_event())

    reranker = reranker_cls.return_value
    query, docs = reranker.call_args.args
    assert query == "what is x?"
    assert len(docs) == 5
    assert reranker.call_args.kwargs == {"top_n": 2}
    assert [d.page_content for d in result.documents] == ["chunk 4", "chunk 3"]


def test_call_without_reranker(make_getter, reranker_cls):
    getter = make_getter(use_reranker=False)
    getter(make_event())
    assert getter.reranker is None
    reranker_cls.assert_not_called()


def test_call_skips_reranker_when_nothing_found(make_getter, qdrant, reranker_cls):
    qdrant.query_points.return_value = SimpleNamespace(points=[])

    result = make_getter(use_reranker=True)(make_event())

    reranker_cls.return_value.assert_not_called()
    assert result.documents == []
    # NO_DOCUMENTS is decided by Generation, retrieval itself succeeded.
    assert result.event.status == ResponseStatus.COMPLETED


@pytest.mark.parametrize(
    ("fail", "code"),
    [
        ("embed", ResponseErrorCode.EMBEDDINGS_ERROR),
        ("qdrant", ResponseErrorCode.QDRANT_ERROR),
    ],
)
def test_call_retrieval_error_marks_event_failed(
    make_getter, embed_dense, qdrant, fail, code
):
    if fail == "embed":
        embed_dense.side_effect = RuntimeError("boom")
    else:
        qdrant.query_points.side_effect = RuntimeError("boom")

    result = make_getter()(make_event())

    assert result.documents == []
    assert result.event.status == ResponseStatus.FAILED
    assert result.event.error_code == code
    assert "boom" in result.event.error_message


def test_call_unexpected_error_is_internal_error(make_getter, qdrant):
    qdrant.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id=1, score=0.1, payload={})]  # no "document"
    )

    result = make_getter()(make_event())

    assert result.documents == []
    assert result.event.status == ResponseStatus.FAILED
    assert result.event.error_code == ResponseErrorCode.INTERNAL_ERROR


def test_call_reranker_exception_is_internal_error(make_getter, reranker_cls):
    reranker_cls.return_value.side_effect = RuntimeError("bug")

    result = make_getter(use_reranker=True)(make_event())

    assert result.documents == []
    assert result.event.error_code == ResponseErrorCode.INTERNAL_ERROR


def test_call_marks_retrieval_span_as_error(make_getter, qdrant):
    qdrant.query_points.side_effect = RuntimeError("boom")
    getter = make_getter()
    getter.langfuse = MagicMock()
    span = getter.langfuse.start_as_current_span.return_value.__enter__.return_value

    event = make_event(tags=["kb"])
    getter(event)

    trace = span.update_trace.call_args.kwargs
    assert trace["name"] == "rag-query"
    assert trace["session_id"] == "7"
    assert trace["tags"] == ["kb"]
    assert trace["input"] == "what is x?"
    assert getter.langfuse.start_as_current_span.call_args.kwargs["trace_context"] == {
        "trace_id": event.trace_id
    }
    span.update.assert_any_call(level="ERROR", status_message=event.error_message)


def test_init_reports_embedding_model_load_error(monkeypatch, make_conf, sent_errors):
    monkeypatch.setattr(
        data_getter, "get_emb_client", MagicMock(side_effect=RuntimeError("no model"))
    )
    with pytest.raises(RuntimeError):
        DataGetter(make_conf())
    assert sent_errors() == ["EMBEDDINGS_ERROR"]
