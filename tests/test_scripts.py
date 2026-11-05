"""Helpers of scripts/load_docs.py and scripts/download_s3.py."""

from pathlib import Path

import httpx
import pytest

import download_s3
import load_docs


# --- load_docs -----------------------------------------------------------------------


@pytest.mark.parametrize("delimiter", [",", ";", "\t"])
def test_read_rows_detects_delimiter(tmp_path, delimiter):
    path = tmp_path / "docs.csv"
    rows = [
        ["name_file", "source", "s3_url", "extra"],
        ["a.md", " src/a ", "s3://bucket/a.md", "x"],
        ["b.md", "src/b", "", "y"],
    ]
    path.write_text(
        "\n".join(delimiter.join(r) for r in rows) + "\n", encoding="utf-8-sig"
    )

    assert load_docs.read_rows(path) == [
        {"name_file": "a.md", "source": "src/a", "s3_url": "s3://bucket/a.md"},
        {"name_file": "b.md", "source": "src/b", "s3_url": ""},
    ]


def test_read_rows_requires_columns(tmp_path):
    path = tmp_path / "docs.csv"
    path.write_text("name_file,url\na.md,s3://bucket/a.md\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="source"):
        load_docs.read_rows(path)


def test_read_dir_relative_sources(tmp_path):
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "docs" / "sub" / "b.txt").write_text("b")

    rows = load_docs.read_dir(tmp_path, [".md", ".txt"], "s3://bucket/")

    assert [(r["name_file"], r["source"]) for r in rows] == [
        ("a.md", "s3://bucket/a.md"),
        ("docs/sub/b.txt", "s3://bucket/docs/sub/b.txt"),
    ]
    assert rows[1]["s3_url"] == (tmp_path / "docs" / "sub" / "b.txt").resolve().as_uri()


def test_read_dir_filters_files(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.md").write_text("x")
    (tmp_path / ".hidden.md").write_text("h")
    (tmp_path / "A.MD").write_text("a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "c.md").mkdir()  # a folder with a matching name

    rows = load_docs.read_dir(tmp_path, ["md"])

    assert [r["source"] for r in rows] == ["A.MD"]


@pytest.mark.parametrize("make_dir", [True, False])
def test_read_dir_no_files(tmp_path, make_dir):
    root = tmp_path / "docs"
    if make_dir:
        root.mkdir()
    with pytest.raises(SystemExit):
        load_docs.read_dir(root, [".md"])


def test_downloader_file(tmp_path):
    path = tmp_path / "Документ 1.md"
    path.write_bytes(b"data")
    assert load_docs.Downloader(verify_ssl=False).get(path.as_uri()) == b"data"


def test_decode_utf8_with_bom():
    assert load_docs.decode("﻿Текст".encode("utf-8")) == "Текст"


def test_decode_cp1251():
    text = "Статья 1. Общие положения настоящего договора и права сторон"
    assert load_docs.decode(text.encode("cp1251")) == text


def test_downloader_http():
    downloader = load_docs.Downloader(verify_ssl=False)
    downloader.http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"data"))
    )
    assert downloader.get("https://files.test/a.md?sig=1") == b"data"


def test_downloader_http_error():
    downloader = load_docs.Downloader(verify_ssl=False)
    downloader.http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(403))
    )
    with pytest.raises(httpx.HTTPStatusError):
        downloader.get("https://files.test/a.md")


def test_downloader_s3():
    class Body:
        def read(self):
            return b"data"

    class S3:
        def get_object(self, Bucket, Key):
            self.args = (Bucket, Key)
            return {"Body": Body()}

    downloader = load_docs.Downloader(verify_ssl=False)
    downloader._s3 = S3()

    assert downloader.get("s3://bucket/dir/a.md") == b"data"
    assert downloader._s3.args == ("bucket", "dir/a.md")


def test_downloader_unsupported_scheme():
    with pytest.raises(ValueError, match="Unsupported"):
        load_docs.Downloader(verify_ssl=False).get("ftp://host/a.md")


# --- download_s3 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "prefix", "expected"),
    [
        ("bucket", "", ("bucket", "")),
        ("bucket", "docs/", ("bucket", "docs/")),
        ("s3://bucket", "", ("bucket", "")),
        ("s3://bucket/docs/", "", ("bucket", "docs/")),
        ("s3://bucket/docs/", "other/", ("bucket", "other/")),
    ],
)
def test_parse_target(target, prefix, expected):
    assert download_s3.parse_target(target, prefix) == expected


def test_local_path_keeps_key_structure(tmp_path):
    out = tmp_path.resolve()
    assert download_s3.local_path(out, "a/b.md") == out / "a" / "b.md"


@pytest.mark.parametrize("key", ["../evil.md", "a/../../evil.md"])
def test_local_path_rejects_keys_outside_out(tmp_path, key):
    with pytest.raises(ValueError, match="outside"):
        download_s3.local_path(tmp_path.resolve(), key)
