from enum import Enum
from typing import Any, Callable

from pydantic import Field, BaseModel
from wunderkafka.time import now
from langchain.docstore.document import Document

ERROR_MESSAGE_LIMIT = 500


class ResponseStatus(str, Enum):
    COMPLETED = 'completed'
    NO_DOCUMENTS = 'no_documents'  # nothing retrieved, LLM not called
    FAILED = 'failed'


class ResponseErrorCode(str, Enum):
    EMBEDDINGS_ERROR = 'EMBEDDINGS_ERROR'
    QDRANT_ERROR = 'QDRANT_ERROR'
    LLM_ERROR = 'LLM_ERROR'
    INTERNAL_ERROR = 'INTERNAL_ERROR'


class MessageEvent(BaseModel):
    message: str
    conversation_id: int
    tags: list[str]
    ts: int = 0
    response: str = ''
    doc_array: list[str] = []
    trace_id: str | None = None
    status: ResponseStatus = ResponseStatus.COMPLETED
    error_code: ResponseErrorCode | None = None
    error_message: str | None = None

    def fail(self, code: ResponseErrorCode, err: Exception | str) -> None:
        self.status = ResponseStatus.FAILED
        self.error_code = code
        self.error_message = str(err)[:ERROR_MESSAGE_LIMIT]

class ResponseEvent(BaseModel):
    """Payload of the backend callback (POST backend_callback_url)."""
    conversation_id: int
    response: str = ''
    doc_array: list[str] = []
    status: ResponseStatus = ResponseStatus.COMPLETED
    error_code: ResponseErrorCode | None = None
    error_message: str | None = None
    trace_id: str | None = None

class RouteExit(BaseModel):
    client: list[BaseModel] | BaseModel | None = None
    tbq: list[BaseModel] | BaseModel | None = None
    task: Callable[[], Any] | None = None

class Response(BaseModel):
    message: str

class Documents(BaseModel):
    query: str
    documents: list[Document] | None
    event: MessageEvent

class RerankedDocuments(BaseModel):
    query: str
    documents: list[Document] | None
    event: MessageEvent

class FinishEvent(MessageEvent):
    ...

class StartEvent(BaseModel):
    ...

class MetricEvent(BaseModel):
    component: str
    measure: str
    ts: int = Field(default_factory=now)
    value: int = Field(default=0)

class MetricKeyEvent(BaseModel):
    ts: int = Field(default_factory=now)
