"""Langfuse tracing for the query path (DataGetter -> Generation).

One user message is one trace. DataGetter and Generation are separate
processors, so the trace id is carried on the event (`MessageEvent.trace_id`).
Tracing is disabled when the Langfuse keys are empty; the SDK then turns
spans and the langfuse.openai wrapper into no-ops.
"""

from config import Configuration
from langchain_core.documents import Document
from langfuse import Langfuse
from loguru import logger

_client: Langfuse | None = None


def init_langfuse(conf: Configuration) -> Langfuse:
    """Create the process-wide client once; get_client(), @observe and
    langfuse.openai reuse it."""
    global _client
    if _client is None:
        public_key = getattr(conf, "langfuse_public_key", "")
        secret_key = getattr(conf, "langfuse_secret_key", "")
        enabled = bool(public_key and secret_key)
        # Placeholder keys still register the (disabled) client, so later
        # get_client() calls reuse it instead of warning about missing keys.
        _client = Langfuse(
            public_key=public_key or "disabled",
            secret_key=secret_key or "disabled",
            host=getattr(conf, "langfuse_host", "") or None,
            environment=getattr(conf, "langfuse_environment", "") or None,
            tracing_enabled=enabled,
        )
        logger.info(f"Langfuse tracing {'enabled' if enabled else 'disabled'}")
    return _client


def new_trace_id() -> str:
    return Langfuse.create_trace_id()


def set_trace_id(event, trace_id: str) -> None:
    # A failure here must not break the request; the generator then starts its own trace.
    try:
        event.trace_id = trace_id
    except (AttributeError, ValueError) as err:
        logger.warning(f"Can't store trace_id on {type(event).__name__}: {err}")


def doc_summary(doc: Document, with_text: bool = False) -> dict:
    """Compact view of a retrieved document for span payloads."""
    keys = ("id", "source", "Header 1", "Header 2", "Header 3", "score", "rerank_score")
    summary = {k: doc.metadata[k] for k in keys if k in doc.metadata}
    if with_text:
        summary["text"] = doc.page_content
    return summary
