"""scripts/run_eval.py: questions CSV -> DataGetter -> Generation -> results CSV + Langfuse run."""

import csv
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langfuse.api import NotFoundError

import run_eval
from src.events import Documents, ResponseErrorCode, ResponseEvent, ResponseStatus


def write_csv(path, rows, delimiter=","):
    path.write_text(
        "\n".join(delimiter.join(r) for r in rows) + "\n", encoding="utf-8-sig"
    )
    return path


class FakeDataGetter:
    def __init__(self) -> None:
        self.events = []

    def __call__(self, event):
        self.events.append(event)
        event.trace_id = f"trace-{event.conversation_id}"
        return Documents(query=event.message, documents=[], event=event)


class FakeGeneration:
    def __call__(self, documents):
        event = documents.event
        if event.message == "boom":
            return ResponseEvent(
                conversation_id=event.conversation_id,
                status=ResponseStatus.FAILED,
                error_code=ResponseErrorCode.LLM_ERROR,
                error_message="llm down",
                trace_id=event.trace_id,
            )
        return ResponseEvent(
            conversation_id=event.conversation_id,
            response=f"answer to {event.message}",
            doc_array=["doc/a", "doc/b"],
            trace_id=event.trace_id,
        )


def fake_langfuse(dataset_exists=True):
    langfuse = MagicMock(name="langfuse")
    if not dataset_exists:
        langfuse.get_dataset.side_effect = NotFoundError(body="not found")
    langfuse.create_dataset_item.side_effect = lambda **kw: SimpleNamespace(id=kw["id"])
    langfuse.get_trace_url.side_effect = lambda trace_id: f"http://lf.test/{trace_id}"
    return langfuse


def make_runner(langfuse, tags=()):
    return run_eval.EvalRunner(
        FakeDataGetter(),
        FakeGeneration(),
        langfuse,
        run_name="run-1",
        run_description="desc",
        metadata={"use_reranker": True},
        tags=list(tags),
    )


def linked(langfuse):
    """(dataset item id, trace id, run name) of each dataset run item created."""
    return [
        (c.kwargs["request"].dataset_item_id, c.kwargs["request"].trace_id, c.kwargs["request"].run_name)
        for c in langfuse.api.dataset_run_items.create.call_args_list
    ]


# --- reading ---------------------------------------------------------------------------


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_read_questions(tmp_path, delimiter):
    path = write_csv(
        tmp_path / "q.csv",
        [["conversation_id", "query", "extra"], ["1", " What is PPE? ", "x"], ["2", "", "y"], ["3", "Who signs?", "z"]],
        delimiter,
    )
    assert run_eval.read_questions(path) == [
        {"conversation_id": "1", "query": "What is PPE?"},
        {"conversation_id": "3", "query": "Who signs?"},
    ]
    assert run_eval.read_questions(path, limit=1) == [
        {"conversation_id": "1", "query": "What is PPE?"}
    ]


def test_read_questions_requires_columns(tmp_path):
    path = write_csv(tmp_path / "q.csv", [["id", "question"], ["1", "q"]])
    with pytest.raises(SystemExit, match="conversation_id"):
        run_eval.read_questions(path)


# --- dataset ---------------------------------------------------------------------------


def test_item_id_is_deterministic_and_per_dataset():
    row = {"conversation_id": "1", "query": "q"}
    assert run_eval.item_id("ds", row) == run_eval.item_id("ds", dict(row))
    assert run_eval.item_id("ds", row) != run_eval.item_id("other", row)
    assert run_eval.item_id("ds", row) != run_eval.item_id("ds", {**row, "query": "q2"})


def test_ensure_dataset_creates_missing_dataset():
    langfuse = fake_langfuse(dataset_exists=False)
    rows = [{"conversation_id": "1", "query": "q1"}, {"conversation_id": "2", "query": "q2"}]

    ids = run_eval.ensure_dataset(langfuse, "ds", rows)

    langfuse.create_dataset.assert_called_once_with(name="ds")
    assert ids == [run_eval.item_id("ds", r) for r in rows]
    assert langfuse.create_dataset_item.call_args_list[0].kwargs["input"] == {
        "query": "q1",
        "conversation_id": "1",
    }


def test_ensure_dataset_keeps_existing_dataset():
    langfuse = fake_langfuse(dataset_exists=True)
    run_eval.ensure_dataset(langfuse, "ds", [{"conversation_id": "1", "query": "q"}])
    langfuse.create_dataset.assert_not_called()


# --- one question ----------------------------------------------------------------------


def test_runner_completed():
    langfuse = fake_langfuse()
    runner = make_runner(langfuse, tags=["kb"])

    result = runner({"conversation_id": "7", "query": "What is PPE?"}, "item-7")

    assert runner.data_getter.events[0].tags == ["kb"]
    assert result["conversation_id"] == "7"
    assert result["response"] == "answer to What is PPE?"
    assert result["status"] == "completed"
    assert result["error_code"] == ""
    assert result["doc_array"] == "doc/a; doc/b"
    assert result["trace_id"] == "trace-7"
    assert result["trace_url"] == "http://lf.test/trace-7"
    assert isinstance(result["latency_s"], float)
    assert linked(langfuse) == [("item-7", "trace-7", "run-1")]
    request = langfuse.api.dataset_run_items.create.call_args.kwargs["request"]
    assert request.metadata == {"use_reranker": True}


def test_runner_pipeline_failure_is_reported_and_linked():
    langfuse = fake_langfuse()
    result = make_runner(langfuse)({"conversation_id": "1", "query": "boom"}, "item-1")

    assert result["status"] == "failed"
    assert result["error_code"] == "LLM_ERROR"
    assert result["error_message"] == "llm down"
    assert result["response"] == ""
    assert linked(langfuse) == [("item-1", "trace-1", "run-1")]


def test_runner_bad_conversation_id():
    langfuse = fake_langfuse()
    result = make_runner(langfuse)({"conversation_id": "abc", "query": "q"}, "item-1")

    assert result["status"] == "failed"
    assert result["error_code"] == "INTERNAL_ERROR"
    assert result["trace_id"] == ""
    assert linked(langfuse) == []


def test_runner_link_and_url_failures_are_not_fatal():
    langfuse = fake_langfuse()
    langfuse.api.dataset_run_items.create.side_effect = RuntimeError("api down")
    langfuse.get_trace_url.side_effect = RuntimeError("api down")

    result = make_runner(langfuse)({"conversation_id": "1", "query": "q"}, "item-1")

    assert result["status"] == "completed"
    assert result["trace_url"] == ""


# --- main ------------------------------------------------------------------------------


@pytest.fixture
def patched(monkeypatch, make_conf):
    langfuse = fake_langfuse(dataset_exists=False)
    conf = make_conf(langfuse_public_key="pk", langfuse_secret_key="sk")
    monkeypatch.setattr(run_eval, "Configuration", lambda: conf)
    monkeypatch.setattr(run_eval, "init_langfuse", lambda c: langfuse)
    monkeypatch.setattr(run_eval, "DataGetter", lambda c: FakeDataGetter())
    monkeypatch.setattr(run_eval, "Generation", lambda c: FakeGeneration())
    return SimpleNamespace(langfuse=langfuse, conf=conf)


def test_main_writes_results_and_links_run(tmp_path, patched):
    questions = write_csv(
        tmp_path / "q.csv",
        [["conversation_id", "query"], ["1", "Что такое СИЗ?"], ["2", "boom"], ["x", "q"]],
    )
    out = tmp_path / "out.csv"

    run_eval.main([str(questions), "--dataset", "ds", "--run-name", "r1", "--out", str(out)])

    with out.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["status"] for r in rows] == ["completed", "failed", "failed"]
    assert rows[0]["response"] == "answer to Что такое СИЗ?"
    assert rows[1]["error_code"] == "LLM_ERROR"
    assert rows[2]["error_code"] == "INTERNAL_ERROR"
    assert list(rows[0]) == run_eval.OUTPUT_COLUMNS
    assert patched.conf.langfuse_environment == "eval"
    assert [(t, r) for _, t, r in linked(patched.langfuse)] == [("trace-1", "r1"), ("trace-2", "r1")]
    patched.langfuse.flush.assert_called_once()


def test_main_default_output_path_and_limit(tmp_path, patched):
    questions = write_csv(
        tmp_path / "q.csv", [["conversation_id", "query"], ["1", "a"], ["2", "b"]]
    )

    run_eval.main([str(questions), "--dataset", "ds", "--run-name", "r1", "--limit", "1"])

    out = tmp_path / "q_results_r1.csv"
    with out.open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_main_flushes_on_error(tmp_path, patched):
    patched.langfuse.create_dataset_item.side_effect = RuntimeError("api down")
    questions = write_csv(tmp_path / "q.csv", [["conversation_id", "query"], ["1", "a"]])

    with pytest.raises(RuntimeError):
        run_eval.main([str(questions), "--dataset", "ds"])
    patched.langfuse.flush.assert_called_once()


def test_main_requires_langfuse_keys(tmp_path, monkeypatch, make_conf):
    monkeypatch.setattr(run_eval, "Configuration", lambda: make_conf())
    questions = write_csv(tmp_path / "q.csv", [["conversation_id", "query"], ["1", "a"]])

    with pytest.raises(SystemExit, match="Langfuse keys"):
        run_eval.main([str(questions), "--dataset", "ds"])


def test_read_questions_with_commas_and_quotes_in_text(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text(
        "conversation_id,query\n"
        '1,"Какие СИЗ нужны на высоте, если высота больше 1,8 м?"\n'
        "2,Кто выдает наряд-допуск; кто его подписывает?\n"
        '3,"Что такое ""опасная зона""?"\n',
        encoding="utf-8",
    )
    assert [q["query"] for q in run_eval.read_questions(path)] == [
        "Какие СИЗ нужны на высоте, если высота больше 1,8 м?",
        "Кто выдает наряд-допуск; кто его подписывает?",
        'Что такое "опасная зона"?',
    ]
