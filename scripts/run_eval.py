"""Run the questions from a CSV through the RAG pipeline and save the answers.

CSV columns: conversation_id, query (other columns are ignored).
Each question goes through the same processors and configuration as in the
service: DataGetter (hybrid search, reranker if use_reranker) and Generation
(LLM). The backend callback is skipped.

Every question is a regular "rag-query" Langfuse trace. The questions are kept
as items of a Langfuse dataset (created if it doesn't exist), and the traces of
one script run are linked to them as a dataset run, so runs can be compared and
scored in Langfuse (Datasets -> <dataset> -> Runs). The Langfuse keys must be set.

Usage (inside the app container, from the project root):
    python scripts/run_eval.py questions.csv --dataset safety-qa
    python scripts/run_eval.py questions.csv --dataset safety-qa --run-name rerank-off --limit 5
"""

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Configuration
from langfuse import Langfuse
from langfuse.api import CreateDatasetRunItemRequest, NotFoundError
from loguru import logger
from src.events import (
    ERROR_MESSAGE_LIMIT,
    MessageEvent,
    ResponseErrorCode,
    ResponseStatus,
)
from src.processors.data_getter import DataGetter
from src.processors.generation_processor import Generation
from src.tracing import init_langfuse

INPUT_COLUMNS = {"conversation_id", "query"}
OUTPUT_COLUMNS = [
    "conversation_id",
    "query",
    "response",
    "status",
    "error_code",
    "error_message",
    "doc_array",
    "trace_id",
    "trace_url",
    "latency_s",
]
# Stored as the dataset run metadata, so runs with different settings can be told apart.
RUN_CONFIG_FIELDS = (
    "use_reranker",
    "k_documents",
    "g_documents",
    "prefetch_k",
    "llm_model_name",
    "embedder_name",
    "reranker_name",
    "qdrant_collection",
)


def read_questions(csv_path: Path, limit: int | None = None) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        # The delimiter is detected from the header only: commas inside quoted
        # questions can make the sniffer fail on the rows.
        header = f.readline()
        f.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(header, delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        missing = INPUT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV has no columns: {sorted(missing)}")
        rows = [{k: (row[k] or "").strip() for k in INPUT_COLUMNS} for row in reader]
    questions = [row for row in rows if row["query"]]
    if len(questions) < len(rows):
        logger.warning(f"{len(rows) - len(questions)} rows with an empty query skipped")
    return questions[:limit] if limit else questions


def item_id(dataset: str, row: dict) -> str:
    """Deterministic, so re-running the same CSV doesn't duplicate dataset items.
    Item ids are global in Langfuse, hence the dataset name in it."""
    return str(uuid5(NAMESPACE_URL, f"{dataset}\n{row['conversation_id']}\n{row['query']}"))


def ensure_dataset(langfuse: Langfuse, name: str, rows: list[dict]) -> list[str]:
    """Create the dataset if needed and upsert one item per row; returns the item ids."""
    try:
        langfuse.get_dataset(name)
    except NotFoundError:
        logger.info(f"Creating Langfuse dataset '{name}'")
        langfuse.create_dataset(name=name)
    ids = []
    for row in rows:
        item = langfuse.create_dataset_item(
            dataset_name=name,
            input={"query": row["query"], "conversation_id": row["conversation_id"]},
            id=item_id(name, row),
        )
        ids.append(item.id)
    return ids


def run_metadata(conf: Configuration, tags: list[str]) -> dict:
    meta = {field: getattr(conf, field, None) for field in RUN_CONFIG_FIELDS}
    meta["tags"] = tags
    return meta


class EvalRunner:
    """Runs one question through DataGetter -> Generation and links its trace to the run."""

    def __init__(
        self,
        data_getter: DataGetter,
        generation: Generation,
        langfuse: Langfuse,
        run_name: str,
        run_description: str,
        metadata: dict,
        tags: list[str],
    ) -> None:
        self.data_getter = data_getter
        self.generation = generation
        self.langfuse = langfuse
        self.run_name = run_name
        self.run_description = run_description
        self.metadata = metadata
        self.tags = tags

    def __call__(self, row: dict, dataset_item_id: str) -> dict:
        result = dict.fromkeys(OUTPUT_COLUMNS, "")
        result.update(conversation_id=row["conversation_id"], query=row["query"])
        start = time.perf_counter()
        # DataGetter and Generation report pipeline errors in the response status;
        # this catches the rest (e.g. a non-integer conversation_id), so the run goes on.
        try:
            event = MessageEvent(
                message=row["query"],
                conversation_id=int(row["conversation_id"]),
                tags=self.tags,
            )
            response = self.generation(self.data_getter(event))
        except Exception as err:
            logger.exception(f"Question failed: {err}")
            result.update(
                status=ResponseStatus.FAILED.value,
                error_code=ResponseErrorCode.INTERNAL_ERROR.value,
                error_message=str(err)[:ERROR_MESSAGE_LIMIT],
                latency_s=round(time.perf_counter() - start, 2),
            )
            return result
        result.update(
            response=response.response,
            status=response.status.value,
            error_code=response.error_code.value if response.error_code else "",
            error_message=response.error_message or "",
            doc_array="; ".join(response.doc_array),
            trace_id=response.trace_id or "",
            latency_s=round(time.perf_counter() - start, 2),
        )
        if response.trace_id:
            result["trace_url"] = self.trace_url(response.trace_id)
            self.link(dataset_item_id, response.trace_id)
        return result

    def trace_url(self, trace_id: str) -> str:
        try:
            return self.langfuse.get_trace_url(trace_id=trace_id) or ""
        except Exception as err:
            logger.warning(f"Can't get the trace url: {err}")
            return ""

    def link(self, dataset_item_id: str, trace_id: str) -> None:
        try:
            self.langfuse.api.dataset_run_items.create(
                request=CreateDatasetRunItemRequest(
                    runName=self.run_name,
                    runDescription=self.run_description,
                    metadata=self.metadata,
                    datasetItemId=dataset_item_id,
                    traceId=trace_id,
                )
            )
        except Exception as err:
            logger.warning(f"Can't link trace {trace_id} to the dataset run: {err}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dataset", required=True, help="Langfuse dataset name")
    parser.add_argument(
        "--run-name", help="Langfuse dataset run name (default: eval-<timestamp>)"
    )
    parser.add_argument(
        "--out", type=Path, help="output CSV (default: <csv>_results_<run name>.csv)"
    )
    parser.add_argument(
        "--tags", nargs="*", default=[], help="retrieval tags for all questions"
    )
    parser.add_argument(
        "--environment", default="eval", help="Langfuse environment of the traces"
    )
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    conf = Configuration()
    if not (
        getattr(conf, "langfuse_public_key", "")
        and getattr(conf, "langfuse_secret_key", "")
    ):
        raise SystemExit(
            "Langfuse keys are empty: set langfuse_public_key and langfuse_secret_key"
        )
    conf.langfuse_environment = args.environment

    rows = read_questions(args.csv, args.limit)
    if not rows:
        raise SystemExit(f"No questions in {args.csv}")
    run_name = args.run_name or f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    out = args.out or args.csv.with_name(f"{args.csv.stem}_results_{run_name}.csv")

    langfuse = init_langfuse(conf)
    data_getter = DataGetter(conf)
    generation = Generation(conf)
    runner = EvalRunner(
        data_getter,
        generation,
        langfuse,
        run_name=run_name,
        run_description=f"{len(rows)} questions from {args.csv.name}",
        metadata=run_metadata(conf, args.tags),
        tags=args.tags,
    )
    logger.info(f"Dataset '{args.dataset}', run '{run_name}', {len(rows)} questions")

    results = []
    try:
        item_ids = ensure_dataset(langfuse, args.dataset, rows)
        # Written row by row, so a crash keeps the answers so far. BOM for Excel.
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for n, (row, dataset_item_id) in enumerate(zip(rows, item_ids), 1):
                result = runner(row, dataset_item_id)
                writer.writerow(result)
                f.flush()
                results.append(result)
                logger.info(
                    f"[{n}/{len(rows)}] {result['status']} in {result['latency_s']}s"
                )
    finally:
        # Spans are exported in the background; without this the last traces are lost.
        langfuse.flush()

    statuses = Counter(r["status"] for r in results)
    latency = sum(r["latency_s"] for r in results) / len(results)
    logger.info(
        f"Done: {dict(statuses)}, avg latency {latency:.2f}s; "
        f"results in {out}, Langfuse run '{run_name}' of dataset '{args.dataset}'"
    )


if __name__ == "__main__":
    main()
