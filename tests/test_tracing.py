import re
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from pydantic import BaseModel

from src import tracing
from src.events import MessageEvent


@pytest.fixture
def fake_langfuse(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    langfuse_cls = MagicMock(name="Langfuse")
    monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)
    return langfuse_cls


def test_init_langfuse_disabled_without_keys(fake_langfuse, conf):
    tracing.init_langfuse(conf)
    kwargs = fake_langfuse.call_args.kwargs
    assert kwargs["tracing_enabled"] is False
    # Placeholder keys keep get_client() from warning about a missing client.
    assert kwargs["public_key"] == "disabled"
    assert kwargs["secret_key"] == "disabled"
    assert kwargs["host"] is None
    assert kwargs["environment"] is None


def test_init_langfuse_enabled_with_keys(fake_langfuse, make_conf):
    conf = make_conf(
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        langfuse_host="https://langfuse.test",
        langfuse_environment="test",
    )
    tracing.init_langfuse(conf)
    kwargs = fake_langfuse.call_args.kwargs
    assert kwargs["tracing_enabled"] is True
    assert (kwargs["public_key"], kwargs["secret_key"]) == ("pk", "sk")
    assert kwargs["host"] == "https://langfuse.test"
    assert kwargs["environment"] == "test"


def test_init_langfuse_needs_both_keys(fake_langfuse, make_conf):
    tracing.init_langfuse(make_conf(langfuse_public_key="pk"))
    assert fake_langfuse.call_args.kwargs["tracing_enabled"] is False


def test_init_langfuse_creates_client_once(fake_langfuse, conf):
    first = tracing.init_langfuse(conf)
    second = tracing.init_langfuse(conf)
    assert first is second
    fake_langfuse.assert_called_once()


def test_new_trace_id_is_w3c_trace_id():
    trace_id = tracing.new_trace_id()
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)
    assert trace_id != tracing.new_trace_id()


def test_set_trace_id_on_message_event():
    event = MessageEvent(message="q", conversation_id=1, tags=[])
    tracing.set_trace_id(event, "abc")
    assert event.trace_id == "abc"


def test_set_trace_id_without_field_does_not_raise():
    class OldEvent(BaseModel):
        message: str

    event = OldEvent(message="q")
    tracing.set_trace_id(event, "abc")  # logs a warning instead
    assert not hasattr(event, "trace_id")


def test_doc_summary_keeps_known_keys_only():
    doc = Document(
        page_content="text",
        metadata={"id": 1, "source": "a.md", "Header 1": "H", "score": 0.5, "tokens": 10},
    )
    assert tracing.doc_summary(doc) == {
        "id": 1,
        "source": "a.md",
        "Header 1": "H",
        "score": 0.5,
    }


def test_doc_summary_with_text():
    doc = Document(page_content="text", metadata={"source": "a.md"})
    assert tracing.doc_summary(doc, with_text=True) == {"source": "a.md", "text": "text"}
