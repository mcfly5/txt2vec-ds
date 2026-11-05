"""Load documents listed in a CSV or found in a local folder into a (new) Qdrant collection.

CSV columns: name_file, source, s3_url.
Each file is downloaded from s3_url and indexed with Txt2Vec, so chunking,
embeddings and point ids are the same as in the service. The file content must
be markdown or plain text; the encoding is detected if it isn't UTF-8
(e.g. cp1251). name_file is only used in logs.

With --dir, every file under the folder with one of the --ext extensions
(.md and .txt by default) is loaded instead; hidden files and folders are
skipped. The source is the path relative to the folder (docs/a.md), with
--source-prefix prepended if given.

The dense vector size is taken from the embedder itself, so the new collection
always matches the model.

Usage (inside the app container, from the project root):
    python scripts/load_docs.py docs.csv --collection documents_v2
    python scripts/load_docs.py docs.csv --collection documents_v2 --tags kb --skip-existing
    python scripts/load_docs.py --dir s3_dump --collection documents_v2 --source-prefix s3://my-bucket/

s3:// urls are read with boto3; credentials and endpoint come from the usual
AWS_* env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL).
http(s):// urls (e.g. presigned) are downloaded directly, file:// urls are
read from the local disk.
"""

import argparse
import csv
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from config import Configuration
from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue
from src.processors.txt2vec import Txt2Vec
from src.processors.vector_store import DENSE_VECTOR, embed_dense, get_emb_client

REQUIRED_COLUMNS = {"name_file", "source", "s3_url"}


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        # The delimiter is detected from the header only: free text in the rows
        # (commas inside quoted values) can make the sniffer fail.
        header = f.readline()
        f.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(header, delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV has no columns: {sorted(missing)}")
        return [
            {k: (v or "").strip() for k, v in row.items() if k in REQUIRED_COLUMNS}
            for row in reader
        ]


def read_dir(root: Path, exts: list[str], source_prefix: str = "") -> list[dict]:
    """Rows in the read_rows format for the files under root."""
    if not root.is_dir():
        raise SystemExit(f"Not a folder: {root}")
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    rows = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rows.append(
            {
                "name_file": rel.as_posix(),
                "source": source_prefix + rel.as_posix(),
                "s3_url": path.resolve().as_uri(),
            }
        )
    if not rows:
        raise SystemExit(f"No {sorted(exts)} files in {root}")
    return rows


class Downloader:
    def __init__(self, verify_ssl: bool) -> None:
        self.http = httpx.Client(verify=verify_ssl, timeout=60, follow_redirects=True)
        self._s3 = None

    @property
    def s3(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3")
        return self._s3

    def get(self, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme == "s3":
            obj = self.s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            return obj["Body"].read()
        if parsed.scheme in ("http", "https"):
            response = self.http.get(url)
            response.raise_for_status()
            return response.content
        if parsed.scheme == "file":
            return Path(url2pathname(parsed.path)).read_bytes()
        raise ValueError(f"Unsupported url scheme: {url}")


def decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    from charset_normalizer import from_bytes

    best = from_bytes(data).best()
    if best is None:
        raise ValueError("Can't detect the text encoding (not a text file?)")
    logger.info(f"Decoded as {best.encoding}")
    return str(best)


def source_exists(txt2vec: Txt2Vec, source: str) -> bool:
    result = txt2vec.client.count(
        collection_name=txt2vec.conf.qdrant_collection,
        count_filter=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
        ),
        exact=True,
    )
    return result.count > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("csv", type=Path, nargs="?", help="CSV with the documents")
    source.add_argument("--dir", type=Path, help="load all files from this folder instead")
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".md", ".txt"],
        help="file extensions to load with --dir",
    )
    parser.add_argument(
        "--source-prefix", default="", help="prepended to the relative path with --dir"
    )
    parser.add_argument("--collection", required=True, help="target collection")
    parser.add_argument("--tags", nargs="*", default=[], help="tags for all documents")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip sources that already have points in the collection",
    )
    parser.add_argument("--verify-ssl", action="store_true", help="verify TLS for downloads")
    args = parser.parse_args()

    conf = Configuration()
    conf.qdrant_collection = args.collection

    emb_size = len(embed_dense(get_emb_client(conf), conf, ["test"])[0])
    if emb_size != conf.emb_size:
        logger.warning(f"emb_size in config is {conf.emb_size}, model returns {emb_size}")
    conf.emb_size = emb_size
    logger.info(f"Collection '{args.collection}', dense size {emb_size}")

    txt2vec = Txt2Vec(conf)  # creates the collection if it doesn't exist
    info = txt2vec.client.get_collection(args.collection)
    existing_size = info.config.params.vectors[DENSE_VECTOR].size
    if existing_size != emb_size:
        raise SystemExit(
            f"Collection '{args.collection}' already exists with size {existing_size}, "
            f"model returns {emb_size}. Use another collection name."
        )

    if args.dir:
        rows = read_dir(args.dir, args.ext, args.source_prefix)
    else:
        rows = read_rows(args.csv)
    downloader = Downloader(args.verify_ssl)
    loaded, skipped, failed = 0, 0, []

    for n, row in enumerate(rows, 1):
        source, url = row["source"], row["s3_url"]
        prefix = f"[{n}/{len(rows)}] {row['name_file']}"
        if not source or not url:
            logger.warning(f"{prefix}: empty source or s3_url, skipped")
            failed.append((row["name_file"], "empty source or s3_url"))
            continue
        if args.skip_existing and source_exists(txt2vec, source):
            logger.info(f"{prefix}: already loaded, skipped")
            skipped += 1
            continue
        try:
            text = decode(downloader.get(url))
            txt2vec._upsert_document(text, source, args.tags)
            loaded += 1
            logger.info(f"{prefix}: loaded")
        except Exception as err:
            logger.error(f"{prefix}: {err!r}")
            failed.append((row["name_file"], repr(err)))

    points = txt2vec.client.count(args.collection, exact=True).count
    logger.info(
        f"Done: loaded {loaded}, skipped {skipped}, failed {len(failed)}; "
        f"{points} points in '{args.collection}'"
    )
    for name, err in failed:
        logger.info(f"  failed: {name}: {err}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
