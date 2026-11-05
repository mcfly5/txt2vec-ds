"""Download all files from an S3 bucket (or a prefix in it) to a local folder.

The key structure is kept: s3://bucket/a/b.md -> <out>/a/b.md (names over 200
bytes are shortened, see short_name). Files that are
already downloaded with the same size are skipped, so the script can be re-run.

Usage (from the project root):
    python scripts/download_s3.py my-bucket
    python scripts/download_s3.py my-bucket --prefix docs/ --out ./s3_dump
    python scripts/download_s3.py s3://my-bucket/docs/ --insecure

Credentials and endpoint come from the usual AWS_* env vars
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL). They are also
read from a .env file (searched from this script's folder upwards); variables
already set in the environment take precedence.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import urllib3
from dotenv import load_dotenv
from loguru import logger

MAX_NAME_BYTES = 200


def parse_target(target: str, prefix: str) -> tuple[str, str]:
    parsed = urlparse(target)
    if parsed.scheme == "s3":
        return parsed.netloc, prefix or parsed.path.lstrip("/")
    return target, prefix


def short_name(name: str) -> str:
    """Fit a path part into the filesystem limit (255 bytes, not chars).

    Long names are cut, keep their extension and get a hash of the full name, so
    they stay unique and map to the same file on every run. The margin leaves
    room for the temp suffix boto3 adds while downloading (".xxxxxxxx").
    """
    if len(name.encode()) <= MAX_NAME_BYTES:
        return name
    suffix = Path(name).suffix
    if len(suffix.encode()) > 20:
        suffix = ""
    tail = "_" + hashlib.sha1(name.encode()).hexdigest()[:8] + suffix
    stem = name[: len(name) - len(suffix)] if suffix else name
    stem = stem.encode()[: MAX_NAME_BYTES - len(tail.encode())].decode(errors="ignore")
    return stem + tail


def local_path(out: Path, key: str) -> Path:
    path = out.joinpath(*(short_name(part) for part in key.split("/") if part)).resolve()
    if not path.is_relative_to(out):
        raise ValueError(f"Key points outside the output folder: {key}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bucket", help="bucket name or s3://bucket/prefix")
    parser.add_argument("--prefix", default="", help="download only keys with this prefix")
    parser.add_argument("--out", type=Path, default=Path("s3_dump"), help="output folder")
    parser.add_argument("--insecure", action="store_true", help="don't verify TLS")
    args = parser.parse_args()
    load_dotenv()

    bucket, prefix = parse_target(args.bucket, args.prefix)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.insecure:  # self-signed certs: skip the per-request warning
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s3 = boto3.client("s3", verify=not args.insecure)
    paginator = s3.get_paginator("list_objects_v2")

    downloaded, skipped, failed = 0, 0, []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key, size = obj["Key"], obj["Size"]
            if key.endswith("/"):  # folder marker
                continue
            try:
                path = local_path(out, key)
                if path.exists() and path.stat().st_size == size:
                    skipped += 1
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(path))
                downloaded += 1
                logger.info(f"{key} ({size} bytes)")
                if path.name != key.rsplit("/", 1)[-1]:
                    logger.warning(f"  name too long, saved as {path.relative_to(out)}")
            except Exception as err:
                logger.error(f"{key}: {err!r}")
                failed.append((key, repr(err)))

    logger.info(
        f"Done: downloaded {downloaded}, skipped {skipped}, failed {len(failed)}; "
        f"files in {out}"
    )
    for key, err in failed:
        logger.info(f"  failed: {key}: {err}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
