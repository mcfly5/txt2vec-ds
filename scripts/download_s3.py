"""Download all files from an S3 bucket (or a prefix in it) to a local folder.

The key structure is kept: s3://bucket/a/b.md -> <out>/a/b.md. Files that are
already downloaded with the same size are skipped, so the script can be re-run.

Usage (from the project root):
    python scripts/download_s3.py my-bucket
    python scripts/download_s3.py my-bucket --prefix docs/ --out ./s3_dump
    python scripts/download_s3.py s3://my-bucket/docs/ --insecure

Credentials and endpoint come from the usual AWS_* env vars
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL).
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from loguru import logger


def parse_target(target: str, prefix: str) -> tuple[str, str]:
    parsed = urlparse(target)
    if parsed.scheme == "s3":
        return parsed.netloc, prefix or parsed.path.lstrip("/")
    return target, prefix


def local_path(out: Path, key: str) -> Path:
    path = (out / key).resolve()
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

    bucket, prefix = parse_target(args.bucket, args.prefix)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

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
