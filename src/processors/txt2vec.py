import uuid

from blocks import Processor
from config import Configuration
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter
from loguru import logger
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointStruct,
    VectorParams,
    models,
)
from src.events import PreprocessedData, Status
from src.processors.vector_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    embed_dense,
    get_bm25_model,
    get_emb_client,
    get_qdrant_client,
    to_sparse_vector,
)
from src.utils.metrics import ErrorCodes, Measures, metric_sender
from transformers import AutoTokenizer, PreTrainedTokenizerBase

HEADERS_TO_SPLIT = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
UPSERT_BATCH_SIZE = 32


class Txt2Vec(Processor):
    def __init__(self, conf: Configuration) -> None:
        self.conf = conf
        self.client = get_qdrant_client(conf)
        try:
            self.emb_client = get_emb_client(conf)
            self.bm25_model = get_bm25_model(conf)
            logger.info("BM25 model loaded...")
            self.tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer_name)
            logger.info(f"Tokenizer loaded ({conf.tokenizer_name})...")
        except Exception as err:
            logger.error("Can't load embedding models")
            metric_sender.send_metric(
                name=Measures.error_code, value=ErrorCodes.EMBEDDINGS_ERROR
            )
            raise err
        self._ensure_collection()

    def __call__(self, preprocessed_data: PreprocessedData) -> Status:
        logger.info("Text to vector store...")
        try:
            self._upsert_document(
                preprocessed_data.data,
                preprocessed_data.source,
                preprocessed_data.event.tags,
            )
        except Exception as e:
            logger.error(
                f"Error in file processing event:{preprocessed_data.event}, error:{e}"
            )
            return Status(
                status="failed",
                message="Error in file processing",
                event=preprocessed_data.event,
            )
        return Status(
            status="completed",
            message="File processed successfully",
            event=preprocessed_data.event,
        )

    def _ensure_collection(self) -> None:
        collection = self.conf.qdrant_collection
        try:
            if not self.client.collection_exists(collection_name=collection):
                logger.info(f"Create collection: {collection}")
                status = self.client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        DENSE_VECTOR: VectorParams(
                            size=self.conf.emb_size, distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR: models.SparseVectorParams(
                            modifier=models.Modifier.IDF
                        )
                    },
                )
                logger.info(status)
            # Indexes for the fields used in filters (tag search, delete by source).
            for field in ("tags", "metadata.source"):
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        except Exception as err:
            logger.error(f"Can't prepare collection {collection}: {err}")
            metric_sender.send_metric(
                name=Measures.error_code, value=ErrorCodes.QDRANT_ERROR
            )
            raise err

    def split_doc(self, doc: str, tokenizer: PreTrainedTokenizerBase) -> list[Document]:
        chunk_size = 500
        chunker = ExperimentalMarkdownSyntaxTextSplitter(HEADERS_TO_SPLIT)
        recursive = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=chunk_size, chunk_overlap=0
        )
        parts = chunker.split_text(doc)
        chunks = []
        for i, part in enumerate(parts):
            tokens = len(tokenizer.encode(part.page_content))
            part.metadata["tokens"] = tokens
            part.metadata["part#"] = i
            if tokens > chunk_size:
                splits = recursive.split_documents([part])
                for j, split in enumerate(splits):
                    split_tokens = len(tokenizer.encode(split.page_content))
                    split.metadata["tokens"] = split_tokens
                    split.metadata["child#"] = j
                    chunks.append(split)
            else:
                chunks.append(part)
        return chunks

    @staticmethod
    def _text_to_embed(chunk: Document) -> str:
        # The markdown splitter strips headers from the content; put the section
        # path back so both dense and BM25 vectors see the section context.
        headers = [
            chunk.metadata[h] for _, h in HEADERS_TO_SPLIT if h in chunk.metadata
        ]
        if not headers:
            return chunk.page_content
        return " > ".join(headers) + "\n\n" + chunk.page_content

    def _upsert_document(self, doc: str, source: str, tags: list[str]) -> None:
        chunks = self.split_doc(doc, self.tokenizer)
        if not chunks:
            logger.warning(f"No chunks produced for source '{source}', skipping")
            return
        texts = [self._text_to_embed(chunk) for chunk in chunks]

        try:
            bm25_embeds = list(self.bm25_model.embed(texts))
            dense_embeds = embed_dense(self.emb_client, self.conf, texts)
        except Exception as err:
            logger.error(
                f"Error while embedding chunks via {self.conf.emb_base_url}: "
                f"{err!r}, cause: {err.__cause__!r}"
            )
            metric_sender.send_metric(
                name=Measures.error_code, value=ErrorCodes.EMBEDDINGS_ERROR
            )
            raise err

        if not len(chunks) == len(bm25_embeds) == len(dense_embeds):
            raise ValueError(
                f"Chunks/embeddings count mismatch: {len(chunks)} chunks, "
                f"{len(bm25_embeds)} sparse, {len(dense_embeds)} dense"
            )

        points = []
        for i, (dense_embed, bm25_embed, chunk) in enumerate(
            zip(dense_embeds, bm25_embeds, chunks)
        ):
            chunk.metadata["source"] = source
            points.append(
                PointStruct(
                    # Deterministic id: re-ingesting a source overwrites its points.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}")),
                    vector={
                        DENSE_VECTOR: dense_embed,
                        SPARSE_VECTOR: to_sparse_vector(bm25_embed),
                    },
                    payload={
                        "document": chunk.page_content,
                        "metadata": chunk.metadata,
                        "tags": tags,
                    },
                )
            )

        try:
            for start in range(0, len(points), UPSERT_BATCH_SIZE):
                self._upsert_batch(points[start : start + UPSERT_BATCH_SIZE])
            # Drop points of an older version of this source that were not overwritten.
            self.client.delete(
                collection_name=self.conf.qdrant_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source", match=MatchValue(value=source)
                        )
                    ],
                    must_not=[HasIdCondition(has_id=[point.id for point in points])],
                ),
            )
        except Exception as err:
            logger.error(f"Error while upserting points: {err}")
            metric_sender.send_metric(
                name=Measures.error_code, value=ErrorCodes.QDRANT_ERROR
            )
            raise err
        logger.info(f"{len(points)} points upserted")

    def _upsert_batch(self, points: list[PointStruct]) -> None:
        # The proxy in front of Qdrant limits the request body size; a dense
        # vector is ~20 KB as JSON, so split the batch until it fits.
        try:
            self.client.upsert(
                collection_name=self.conf.qdrant_collection, points=points
            )
        except UnexpectedResponse as err:
            if err.status_code != 413 or len(points) == 1:
                raise
            logger.warning(f"Batch of {len(points)} points is too large, splitting")
            half = len(points) // 2
            self._upsert_batch(points[:half])
            self._upsert_batch(points[half:])

    def delete_documents_by_source(self, source: str) -> dict:
        """Delete all documents from Qdrant collection with the specified source.

        Args:
            source: Source value to filter documents for deletion

        Returns:
            dict: Response with deletion status and count

        """
        source_filter = Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
        )
        try:
            count_result = self.client.count(
                collection_name=self.conf.qdrant_collection,
                count_filter=source_filter,
                exact=True,
            )

            docs_to_delete = count_result.count
            logger.info(f"Found {docs_to_delete} documents with source '{source}'")

            if docs_to_delete == 0:
                return {
                    "status": "success",
                    "deleted_count": 0,
                    "message": f"No documents found with source '{source}'",
                }

            delete_result = self.client.delete(
                collection_name=self.conf.qdrant_collection,
                points_selector=source_filter,
            )

            logger.info("Document deleted")

            return {
                "status": "success",
                "deleted_count": docs_to_delete,
                "message": f"Successfully deleted {docs_to_delete} documents with source '{source}'",
                "qdrant_response": delete_result,
            }

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return {"status": "error", "message": f"Error deleting documents: {str(e)}"}
