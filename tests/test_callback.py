import json

import pytest
import requests
from requests.adapters import BaseAdapter

from src.events import ResponseErrorCode, ResponseEvent, ResponseStatus
from src.processors.callback import Callback


class FakeAdapter(BaseAdapter):
    """Records the requests and answers with `status` (or raises `error`)."""

    def __init__(self, status: int = 200, error: Exception | None = None) -> None:
        super().__init__()
        self.status = status
        self.error = error
        self.requests: list[requests.PreparedRequest] = []
        self.timeouts: list = []

    def send(self, request, timeout=None, **kwargs):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.error:
            raise self.error
        response = requests.Response()
        response.status_code = self.status
        response._content = b"ok"
        response.url = request.url
        response.request = request
        return response

    def close(self) -> None:
        pass


@pytest.fixture
def make_callback(conf):
    def make(**adapter_kwargs) -> tuple[Callback, FakeAdapter]:
        callback = Callback(conf)
        adapter = FakeAdapter(**adapter_kwargs)
        callback.session.mount("http://", adapter)
        return callback, adapter

    return make


def response_event(**kwargs) -> ResponseEvent:
    return ResponseEvent(conversation_id=7, trace_id="abc", **kwargs)


def test_posts_response_event_as_json(make_callback, conf, sent_errors):
    callback, adapter = make_callback()

    callback(response_event(response="answer", doc_array=["a.md"]))

    (request,) = adapter.requests
    assert request.method == "POST"
    assert request.url == conf.backend_callback_url
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.body) == {
        "conversation_id": 7,
        "response": "answer",
        "doc_array": ["a.md"],
        "status": "completed",
        "error_code": None,
        "error_message": None,
        "trace_id": "abc",
    }
    assert adapter.timeouts == [conf.backend_callback_timeout]
    assert sent_errors() == []


def test_failed_status_is_sent_as_strings(make_callback):
    callback, adapter = make_callback()

    callback(
        response_event(
            status=ResponseStatus.FAILED,
            error_code=ResponseErrorCode.LLM_ERROR,
            error_message="boom",
        )
    )

    body = json.loads(adapter.requests[0].body)
    assert body["status"] == "failed"
    assert body["error_code"] == "LLM_ERROR"
    assert body["error_message"] == "boom"


@pytest.mark.parametrize("status", [400, 500, 503])
def test_http_error_status_is_reported(make_callback, sent_errors, status):
    callback, adapter = make_callback(status=status)

    callback(response_event())  # doesn't raise

    assert len(adapter.requests) == 1  # not retried
    assert sent_errors() == ["BACKEND_ERROR"]


@pytest.mark.parametrize(
    "error", [requests.ConnectionError("refused"), requests.Timeout("slow")]
)
def test_request_error_is_reported(make_callback, sent_errors, error):
    callback, adapter = make_callback(error=error)

    callback(response_event())

    assert sent_errors() == ["BACKEND_ERROR"]
