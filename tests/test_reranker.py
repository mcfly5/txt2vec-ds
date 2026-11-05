import json

import httpx
import pytest
from langchain_core.documents import Document

from src.processors.reranker import Reranker


def make_docs(n: int) -> list[Document]:
    return [Document(page_content=f"doc {i}", metadata={"id": i}) for i in range(n)]


@pytest.fixture
def requests_seen():
    return []


@pytest.fixture
def make_reranker(conf, requests_seen):
    """Reranker whose HTTP client is served by `handler` instead of Infinity."""

    def make(handler, conf=conf) -> Reranker:
        reranker = Reranker(conf)

        def record(request: httpx.Request) -> httpx.Response:
            requests_seen.append(request)
            return handler(request)

        reranker.client = httpx.Client(
            base_url=conf.reranker_base_url,
            headers=reranker.client.headers,
            transport=httpx.MockTransport(record),
        )
        return reranker

    return make


def scores(*pairs) -> callable:
    results = [{"index": i, "relevance_score": s} for i, s in pairs]
    return lambda request: httpx.Response(200, json={"results": results})


def test_reorders_by_relevance_and_stores_score(make_reranker, requests_seen, conf):
    reranker = make_reranker(scores((0, 0.1), (1, 0.9), (2, 0.5)))
    docs = make_docs(3)

    result = reranker("query", docs, top_n=2)

    assert [d.metadata["id"] for d in result] == [1, 2]
    assert [d.metadata["rerank_score"] for d in result] == [0.9, 0.5]

    request = requests_seen[0]
    assert request.method == "POST"
    assert request.url.path == "/rerank"
    assert json.loads(request.content) == {
        "model": conf.reranker_name,
        "query": "query",
        "documents": ["doc 0", "doc 1", "doc 2"],
        "return_documents": False,
        "top_n": 2,
    }


def test_without_top_n_returns_all_and_omits_it_from_request(make_reranker, requests_seen):
    reranker = make_reranker(scores((0, 0.2), (1, 0.3)))

    result = reranker("query", make_docs(2))

    assert [d.metadata["id"] for d in result] == [1, 0]
    assert "top_n" not in json.loads(requests_seen[0].content)


def test_auth_header_only_with_api_key(make_reranker, requests_seen, make_conf):
    make_reranker(scores(), conf=make_conf())("q", [])
    make_reranker(scores(), conf=make_conf(reranker_api_key="secret"))("q", [])

    assert "authorization" not in requests_seen[0].headers
    assert requests_seen[1].headers["authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda r: httpx.Response(500, text="boom"), id="http-500"),
        pytest.param(lambda r: httpx.Response(200, json={"error": "x"}), id="bad-body"),
        pytest.param(
            lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")), id="connect"
        ),
    ],
)
def test_failure_keeps_retrieval_order(make_reranker, sent_errors, handler):
    reranker = make_reranker(handler)
    docs = make_docs(5)

    result = reranker("query", docs, top_n=3)

    assert [d.metadata["id"] for d in result] == [0, 1, 2]
    assert all("rerank_score" not in d.metadata for d in result)
    assert sent_errors() == ["RERANKER_ERROR"]


def test_client_settings(make_conf):
    reranker = Reranker(make_conf(reranker_timeout=7))
    assert reranker.client.timeout.read == 7
    assert str(reranker.client.base_url).startswith("http://infinity.test")
