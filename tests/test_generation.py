from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.events import Documents, MessageEvent, ResponseErrorCode, ResponseStatus
from src.processors.generation_processor import Generation

TRACE_ID = "0123456789abcdef" * 2


@pytest.fixture
def generation(conf) -> Generation:
    gen = Generation(conf)
    gen.llm = MagicMock(name="llm")
    gen.llm.call.return_value = SimpleNamespace(content="the answer")
    return gen


def make_docs(*sources: str) -> list[Document]:
    return [Document(page_content=f"text of {s}", metadata={"source": s}) for s in sources]


def make_documents(docs, **event) -> Documents:
    event = MessageEvent(
        message="what is x?", conversation_id=7, tags=[], trace_id=TRACE_ID, **event
    )
    return Documents(query=event.message, documents=docs, event=event)


def test_completed(generation):
    response = generation(make_documents(make_docs("a.md", "b.md")))

    assert response.status == ResponseStatus.COMPLETED
    assert response.response == "the answer"
    assert response.doc_array == ["a.md", "b.md"]
    assert response.error_code is None
    assert response.error_message is None
    assert response.conversation_id == 7
    assert response.trace_id == TRACE_ID


def test_prompt_contains_query_and_numbered_documents(generation):
    generation(make_documents(make_docs("a.md", "b.md")))

    prompt = generation.llm.call.call_args.args[0]
    assert "what is x?" in prompt
    assert "Document #:1\ntext of a.md\nDocument #:2\ntext of b.md" in prompt


@pytest.mark.parametrize("docs", [[], None])
def test_no_documents_skips_llm(generation, docs):
    response = generation(make_documents(docs))

    generation.llm.call.assert_not_called()
    assert response.status == ResponseStatus.NO_DOCUMENTS
    assert response.error_code is None
    assert response.response == ""
    assert response.doc_array == []


def test_failed_retrieval_is_passed_through(generation):
    documents = make_documents([])
    documents.event.fail(ResponseErrorCode.QDRANT_ERROR, "qdrant: timeout")

    response = generation(documents)

    generation.llm.call.assert_not_called()
    assert response.status == ResponseStatus.FAILED
    assert response.error_code == ResponseErrorCode.QDRANT_ERROR
    assert response.error_message == "qdrant: timeout"
    assert response.response == ""


def test_llm_exception(generation):
    generation.llm.call.side_effect = RuntimeError("503 from vLLM")

    response = generation(make_documents(make_docs("a.md")))

    assert response.status == ResponseStatus.FAILED
    assert response.error_code == ResponseErrorCode.LLM_ERROR
    assert response.error_message == "503 from vLLM"
    assert response.response == ""
    assert response.doc_array == []


@pytest.mark.parametrize("content", ["", None])
def test_llm_empty_content(generation, content):
    generation.llm.call.return_value = SimpleNamespace(content=content)

    response = generation(make_documents(make_docs("a.md")))

    assert response.status == ResponseStatus.FAILED
    assert response.error_code == ResponseErrorCode.LLM_ERROR
    assert response.error_message == "Empty LLM response"


def test_unexpected_error_is_internal_error(generation):
    # A document without "source" fails after the LLM answered; the answer
    # must not be sent with a failed status.
    docs = [Document(page_content="text", metadata={})]

    response = generation(make_documents(docs))

    assert response.status == ResponseStatus.FAILED
    assert response.error_code == ResponseErrorCode.INTERNAL_ERROR
    assert response.response == ""
    assert response.doc_array == []


@pytest.fixture
def span(generation):
    generation.langfuse = MagicMock()
    return generation.langfuse.start_as_current_span.return_value.__enter__.return_value


def test_joins_trace_from_event(generation, span):
    generation(make_documents(make_docs("a.md")))

    kwargs = generation.langfuse.start_as_current_span.call_args.kwargs
    assert kwargs["name"] == "generation"
    assert kwargs["trace_context"] == {"trace_id": TRACE_ID}
    span.update_trace.assert_called_once_with(output="the answer")


def test_starts_new_trace_without_trace_id(generation, span):
    documents = make_documents(make_docs("a.md"))
    documents.event.trace_id = None

    generation(documents)

    assert generation.langfuse.start_as_current_span.call_args.kwargs["trace_context"] is None


def test_span_level_error_on_failure(generation, span):
    generation.llm.call.side_effect = RuntimeError("boom")

    generation(make_documents(make_docs("a.md")))

    span.update.assert_any_call(level="ERROR", status_message="boom")
    output = span.update.call_args_list[0].kwargs["output"]
    assert output["status"] == ResponseStatus.FAILED
    assert "conversation_id" not in output
    assert "trace_id" not in output


def test_span_not_error_on_success(generation, span):
    generation(make_documents(make_docs("a.md")))
    assert all("level" not in c.kwargs for c in span.update.call_args_list)
