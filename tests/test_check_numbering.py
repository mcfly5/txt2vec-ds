import csv
import sys

import pytest

import check_numbering as cn


def items(text: str) -> list[tuple[str, str]]:
    return [(item.stream, cn.fmt(item.num)) for item in cn.extract_items(text)]


def issues(text: str, max_gap: int = 20) -> list[tuple[str, str, list[str]]]:
    """(got, kind, missing) for every reported number."""
    checkers: dict[str, cn.SequenceChecker] = {}
    found = []
    for item in cn.extract_items(text):
        result = checkers.setdefault(item.stream, cn.SequenceChecker(max_gap)).feed(item)
        if result:
            kind, missing = result
            found.append((cn.fmt(item.num), kind, [cn.fmt(m) for m in missing]))
    return found


def lines(*rows: str) -> str:
    return "\n".join(rows)


# --- extract_items ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("7.6. Text", [("point", "7.6")]),
        ("3) Text", [("point", "3")]),
        ("3. Text", [("point", "3")]),
        ("  2.1.4 Text", [("point", "2.1.4")]),
        ("## 7.6 Title", [("point", "7.6")]),
        ("# 5 Title", [("point", "5")]),
        ("**1.2.** Bold", [("point", "1.2")]),
        ("Статья 7. Name", [("article", "7")]),
        ("Article 12", [("article", "12")]),
        ("Глава 3", [("chapter", "3")]),
        ("Раздел 2", [("section", "2")]),
        ("Часть 1", [("section", "1")]),
        ("п. 2.1 Text", [("point", "2.1")]),
        ("Пункт 4", [("point", "4")]),
    ],
)
def test_extract_numbered_lines(line, expected):
    assert items(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "5 apples",  # a bare number is a quantity, not a point
        "01.02.2020 date",
        "1,5 ratio",
        "7.6/2 fraction",
        "12:30 time",
        "Text with 7.6. inside",
        "| 3. table cell |",
    ],
)
def test_extract_ignores_non_numbering(line):
    assert items(line) == []


def test_extract_ignores_code_blocks():
    text = lines("1. a", "```", "2. code", "```", "2. b")
    assert [i.line for i in cn.extract_items(text)] == [1, 5]


# --- SequenceChecker ------------------------------------------------------------------


def test_consecutive_numbering_has_no_issues():
    assert issues(lines("1. a", "2. b", "2.1. c", "2.2. d", "3. e", "3.1 f")) == []


def test_gap_in_points():
    text = lines("1. a", "1.1. b", "1.2. c", "1.5. d", "1.6. e")
    assert issues(text) == [("1.5", "gap", ["1.3", "1.4"])]


def test_gap_in_top_level():
    assert issues(lines("1. a", "2. b", "5. c")) == [("5", "gap", ["3", "4"])]


def test_nested_list_restart_is_not_reported():
    assert issues(lines("1. a", "2. b", "1) x", "2) y", "3. c")) == []


def test_restart_after_contents_is_not_reported():
    toc = ["1. Intro", "2. Terms", "3. Rules"]
    assert issues(lines(*toc, *toc)) == []


def test_jump_above_max_gap_is_out_of_order():
    assert issues(lines("1. a", "2. b", "50. c", "3. d")) == [("50", "out_of_order", [])]
    assert issues(lines("1. a", "2. b", "6. c"), max_gap=2) == [("6", "out_of_order", [])]


def test_streams_are_checked_separately():
    text = lines("Статья 1", "1. a", "Статья 2", "2. b", "Статья 4", "3. c", "Глава 1")
    assert issues(text) == [("4", "gap", ["3"])]


def test_new_parent_resets_child_numbering():
    assert issues(lines("1. a", "1.1 b", "1.2 c", "2. d", "2.1 e")) == []


# --- check_file / main ------------------------------------------------------------------


def test_check_file_reports_issue_fields(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(lines("# Doc", "1. a", "2. b", "5. c"), encoding="utf-8")

    found, count = cn.check_file(path, "doc.md", max_gap=20)

    assert count == 3
    assert [vars(i) for i in found] == [
        {
            "file": "doc.md",
            "stream": "point",
            "line": 4,
            "prev": "2",
            "got": "5",
            "missing": "3, 4",
            "kind": "gap",
            "text": "5. c",
        }
    ]


def test_check_file_decodes_cp1251(tmp_path):
    path = tmp_path / "doc.txt"
    text = lines("Статья 1. Общие положения", "Статья 2. Права и обязанности сторон")
    path.write_bytes(text.encode("cp1251"))

    found, count = cn.check_file(path, "doc.txt", max_gap=20)

    assert (found, count) == ([], 2)


def run_main(monkeypatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["check_numbering.py", *args])
    try:
        cn.main()
    except SystemExit as exc:
        return exc.code
    return 0


def read_report(path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_main_writes_report_and_fails_on_gaps(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / "ok.md").write_text(lines("1. a", "2. b"), encoding="utf-8")
    (docs / "sub" / "gap.txt").write_text(lines("1. a", "2. b", "4. d"), encoding="utf-8")
    (docs / "skipped.pdf").write_text(lines("1. a", "9. b"), encoding="utf-8")
    report = tmp_path / "report.csv"

    code = run_main(monkeypatch, str(docs), "--report", str(report))

    assert code == 1
    rows = read_report(report)
    assert [(r["file"], r["got"], r["missing"], r["kind"]) for r in rows] == [
        ("sub/gap.txt", "4", "3", "gap")
    ]
    assert list(rows[0]) == list(cn.Issue.__dataclass_fields__)


def test_main_passes_without_gaps(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ok.md").write_text(lines("1. a", "2. b", "50. ref", "3. c"), encoding="utf-8")
    (docs / "plain.md").write_text("no numbering here", encoding="utf-8")
    report = tmp_path / "report.csv"

    code = run_main(monkeypatch, str(docs), "--report", str(report))

    assert code == 0
    assert [r["kind"] for r in read_report(report)] == ["out_of_order"]


def test_main_ext_filter(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "gap.md").write_text(lines("1. a", "3. c"), encoding="utf-8")
    report = tmp_path / "report.csv"

    assert run_main(monkeypatch, str(docs), "--ext", "txt", "--report", str(report)) == 0
    assert read_report(report) == []


def test_main_rejects_missing_folder(tmp_path, monkeypatch):
    code = run_main(monkeypatch, str(tmp_path / "nope"))
    assert "Not a folder" in str(code)
