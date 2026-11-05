"""Stubs for the in-house modules that aren't part of this repo, and shared fixtures.

The service code imports `blocks`, `config` (built on `configuration.DigitalConfig`),
`src.utils` and `wunderkafka`, which come from the application this
repo is deployed into. They are replaced here with minimal stand-ins before any
test module imports the code under test.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


class Processor:
    pass


class OutputUrl:
    def __init__(self, url, event, method="POST", headers=None, cookies=None) -> None:
        self.url = url
        self.event = event
        self.method = method
        self.headers = headers or {}
        self.cookies = cookies or {}


class Configuration:
    """Required fields only; optional ones (prefetch_k, *_timeout, ...) are left
    unset so the getattr defaults in the code are used unless a test sets them."""

    def __init__(self, **overrides) -> None:
        self.llm_model_name = "test-llm"
        self.llm_base_url = "http://llm.test/v1"
        self.llm_api_key = "key"
        self.llm_system_prompt = "You are a helpful assistant."
        self.embedder_name = "test-embedder"
        self.emb_base_url = "http://infinity.test"
        self.emb_api_key = ""
        self.emb_size = 4
        self.tokenizer_name = "test-tokenizer"
        self.reranker_name = "test-reranker"
        self.reranker_base_url = "http://infinity.test"
        self.reranker_api_key = ""
        self.use_reranker = False
        self.k_documents = 100
        self.g_documents = 10
        self.qdrant_url = "https://qdrant.test"
        self.qdrant_collection = "documents"
        self.backend_callback_url = "http://backend.test/api/v1/model_result"
        self.backend_callback_timeout = 10.0
        self.langfuse_host = ""
        self.langfuse_public_key = ""
        self.langfuse_secret_key = ""
        self.langfuse_environment = ""
        self.__dict__.update(overrides)


class Measures:
    error_code = "error_code"


class ErrorCodes:
    EMBEDDINGS_ERROR = "EMBEDDINGS_ERROR"
    QDRANT_ERROR = "QDRANT_ERROR"
    RERANKER_ERROR = "RERANKER_ERROR"
    BACKEND_ERROR = "BACKEND_ERROR"


metric_sender = MagicMock(name="metric_sender")

_module("blocks", Processor=Processor)
_module("blocks.connectors")
_module("blocks.connectors.http")
_module("blocks.connectors.http.events", OutputUrl=OutputUrl)
_module("wunderkafka")
_module("wunderkafka.time", now=lambda: 0)
_module("config", Configuration=Configuration)

import src  # noqa: E402  (namespace package; the stubs below are attached to it)

_module("src.utils")
_utils = dict(Measures=Measures, ErrorCodes=ErrorCodes, metric_sender=metric_sender)
_module("src.utils.utils", **_utils)
_module("src.utils.metrics", **_utils)

# txt2vec.py imports the indexing events, which live in the indexing service's
# src/events.py, not in this one.
import src.events  # noqa: E402


class _InputEvent(BaseModel):
    tags: list[str] = []


class PreprocessedData(BaseModel):
    data: str
    source: str
    event: _InputEvent


class Status(BaseModel):
    status: str
    message: str
    event: _InputEvent


src.events.PreprocessedData = PreprocessedData
src.events.Status = Status
src.events.InputEvent = _InputEvent


@pytest.fixture(autouse=True)
def _reset_metrics():
    metric_sender.reset_mock()
    yield


@pytest.fixture
def sent_errors():
    """Returns the error codes sent with metric_sender so far in the test."""
    return lambda: [
        c.kwargs["value"]
        for c in metric_sender.send_metric.call_args_list
        if c.kwargs.get("name") == Measures.error_code
    ]


@pytest.fixture
def conf() -> Configuration:
    return Configuration()


@pytest.fixture
def make_conf():
    return Configuration


@pytest.fixture
def word_tokenizer():
    """Offline HF tokenizer: one token per word or punctuation mark."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
