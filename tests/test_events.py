from src.events import (
    ERROR_MESSAGE_LIMIT,
    MessageEvent,
    ResponseErrorCode,
    ResponseEvent,
    ResponseStatus,
)


def make_event(**kwargs) -> MessageEvent:
    return MessageEvent(message="question", conversation_id=1, tags=[], **kwargs)


def test_message_event_defaults_to_completed():
    event = make_event()
    assert event.status == ResponseStatus.COMPLETED
    assert event.error_code is None
    assert event.error_message is None
    assert event.trace_id is None


def test_fail_sets_status_code_and_message():
    event = make_event()
    event.fail(ResponseErrorCode.QDRANT_ERROR, RuntimeError("connection refused"))
    assert event.status == ResponseStatus.FAILED
    assert event.error_code == ResponseErrorCode.QDRANT_ERROR
    assert event.error_message == "connection refused"


def test_fail_truncates_long_message():
    event = make_event()
    event.fail(ResponseErrorCode.LLM_ERROR, "x" * (ERROR_MESSAGE_LIMIT + 100))
    assert len(event.error_message) == ERROR_MESSAGE_LIMIT


def test_response_event_serializes_enum_values():
    # The backend gets plain strings, not enum names.
    event = ResponseEvent(
        conversation_id=1,
        status=ResponseStatus.FAILED,
        error_code=ResponseErrorCode.EMBEDDINGS_ERROR,
    )
    data = event.model_dump(mode="json")
    assert data["status"] == "failed"
    assert data["error_code"] == "EMBEDDINGS_ERROR"
    assert data["response"] == ""
    assert data["doc_array"] == []
