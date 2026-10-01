"""Check downloaded documents for gaps in the numbering of articles and points.

The PDF parser sometimes drops parts of a document; this shows up as a jump in
the numbering (e.g. 7.6 is followed by 7.9). The script finds such jumps in all
files of a folder and writes them to a CSV report.

Numbered lines are found at the line start: plain (`7.6.`, `3)`), in markdown
headers (`## 7.6 Title`) or after a keyword (`Статья 7.6`, `Глава 3`, `п. 2.1`).
Articles, chapters, sections and points are checked as separate sequences.

Report kinds:
    gap           numbers are missing, e.g. after 7.6 got 7.9 (missing 7.7, 7.8)
    out_of_order  a number that doesn't continue the sequence (a cross-reference,
                  a stray number or a big jump); worth a look, usually not lost text

Usage (from the project root):
    python scripts/check_numbering.py
    python scripts/check_numbering.py ./s3_dump --report gaps.csv --ext .md .txt

Exits with 1 if any gap is found.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

KEYWORD_STREAMS = {
    "статья": "article",
    "article": "article",
    "глава": "chapter",
    "chapter": "chapter",
    "раздел": "section",
    "section": "section",
    "часть": "section",
    "part": "section",
    "пункт": "point",
    "п.": "point",
}

NUMBERED_LINE = re.compile(
    r"^\s*(?P<header>#{1,6}\s+)?(?:[*_]{1,2})?\s*"
    r"(?:(?P<kw>статья|article|глава|chapter|раздел|section|часть|part|пункт|п\.)\s*)?"
    r"(?P<num>\d{1,3}(?:\.\d{1,3}){0,5})(?P<delim>[.)]?)"
    r"(?![\d.,:/-])",
    re.IGNORECASE,
)

Number = tuple[int, ...]


@dataclass
class Item:
    stream: str
    num: Number
    line: int
    text: str


@dataclass
class Issue:
    file: str
    stream: str
    line: int
    prev: str
    got: str
    missing: str
    kind: str
    text: str


def fmt(num: Number) -> str:
    return ".".join(map(str, num))


def decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    from charset_normalizer import from_bytes

    best = from_bytes(data).best()
    if best is None:
        raise ValueError("Can't detect the text encoding (not a text file?)")
    return str(best)


def extract_items(text: str) -> list[Item]:
    items = []
    in_code = False
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or stripped.startswith("|"):
            continue
        match = NUMBERED_LINE.match(line)
        if not match:
            continue
        num = tuple(int(part) for part in match["num"].split("."))
        kw = (match["kw"] or "").lower()
        # a bare "5 ..." at the line start is usually a quantity, not a point
        if len(num) == 1 and not kw and not match["header"] and not match["delim"]:
            continue
        items.append(Item(KEYWORD_STREAMS.get(kw, "point"), num, line_no, stripped))
    return items


@dataclass
class State:
    # last child number of each multi-level parent, e.g. (7,) -> 6 after "7.6"
    children: dict[Number, int] = field(default_factory=dict)
    # counters of the top-level sequences; nested lists and restarts push a new one
    top: list[int] = field(default_factory=list)

    def copy(self) -> "State":
        return State(dict(self.children), list(self.top))

    def clear_below(self, num: Number) -> None:
        for parent in [p for p in self.children if p[: len(num)] == num]:
            del self.children[parent]


class SequenceChecker:
    """Checks one numbering sequence (stream) of a file."""

    def __init__(self, max_gap: int) -> None:
        self.max_gap = max_gap
        self.state = State()
        # state as if the last out-of-order number started a new numbering;
        # used if the next number continues it
        self.alt: State | None = None
        self.prev = ""

    def feed(self, item: Item) -> tuple[str, list[Number]] | None:
        """Returns (kind, missing numbers) or None if the number is expected."""
        prev, self.prev = self.prev, fmt(item.num)
        kind, missing, new_state = self._eval(self.state, item.num)
        if kind != "out_of_order":
            self.state, self.alt = new_state, None
            return (kind, missing) if kind else None
        if self.alt is not None:
            alt_kind, alt_missing, alt_state = self._eval(self.alt, item.num)
            if alt_kind != "out_of_order":
                self.state, self.alt = alt_state, None
                return (alt_kind, alt_missing) if alt_kind else None
        self.alt = new_state
        return "out_of_order", []

    def _eval(self, state: State, num: Number) -> tuple[str | None, list[Number], State]:
        state = state.copy()
        results = [self._top(state, num[0], implicit=len(num) > 1)]
        for depth in range(2, len(num) + 1):
            results.append(self._child(state, num[:depth], implicit=depth < len(num)))
        kinds = {kind for kind, _ in results}
        missing = [m for _, found in results for m in found]
        if "out_of_order" in kinds:
            return "out_of_order", [], state
        return ("gap" if missing else None), missing, state

    def _classify(self, missing: list[Number]) -> tuple[str | None, list[Number]]:
        if len(missing) > self.max_gap:
            return "out_of_order", []
        return ("gap" if missing else None), missing

    def _top(self, state: State, k: int, implicit: bool) -> tuple[str | None, list[Number]]:
        stack = state.top
        if implicit and k in stack:  # "7.2" inside chapter 7
            del stack[len(stack) - stack[::-1].index(k) :]
            return None, []
        state.clear_below((k,))
        if k == 1:  # a new list, or a restart (e.g. the body after the contents)
            stack.append(1)
            return None, []
        if not stack:
            stack.append(k)
            return self._classify([(n,) for n in range(1, k)])
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] + 1 == k:
                del stack[i + 1 :]
                stack[i] = k
                return None, []
        below = [(c, i) for i, c in enumerate(stack) if c < k]
        if below:
            c, i = max(below)
            del stack[i + 1 :]
            stack[i] = k
            return self._classify([(n,) for n in range(c + 1, k)])
        stack.append(k)
        return "out_of_order", []

    def _child(self, state: State, num: Number, implicit: bool) -> tuple[str | None, list[Number]]:
        parent, k = num[:-1], num[-1]
        last = state.children.get(parent, 0)
        if implicit and k == last:
            return None, []
        state.children[parent] = k
        state.clear_below(num)
        if k == last + 1 or k == 1:  # next one, or a restart (e.g. body after the contents)
            return None, []
        if k > last:
            return self._classify([parent + (n,) for n in range(last + 1, k)])
        return "out_of_order", []


def check_file(path: Path, name: str, max_gap: int) -> tuple[list[Issue], int]:
    """Returns the issues and the number of numbered items found."""
    items = extract_items(decode(path.read_bytes()))
    checkers: dict[str, SequenceChecker] = {}
    issues = []
    for item in items:
        checker = checkers.setdefault(item.stream, SequenceChecker(max_gap))
        prev = checker.prev or "start"
        result = checker.feed(item)
        if result is None:
            continue
        kind, missing = result
        issues.append(
            Issue(
                file=name,
                stream=item.stream,
                line=item.line,
                prev=prev,
                got=fmt(item.num),
                missing=", ".join(map(fmt, missing)),
                kind=kind,
                text=item.text[:120],
            )
        )
    return issues, len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", type=Path, nargs="?", default=Path("s3_dump"))
    parser.add_argument("--ext", nargs="+", default=[".md", ".txt"], help="file extensions")
    parser.add_argument("--report", type=Path, default=Path("numbering_report.csv"))
    parser.add_argument(
        "--max-gap",
        type=int,
        default=20,
        help="jumps with more missing numbers are reported as out_of_order",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Not a folder: {args.folder}")
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}
    files = sorted(p for p in args.folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)

    all_issues: list[Issue] = []
    unnumbered, failed = [], []
    for path in files:
        name = path.relative_to(args.folder).as_posix()
        try:
            issues, count = check_file(path, name, args.max_gap)
        except Exception as err:
            logger.error(f"{name}: {err!r}")
            failed.append(name)
            continue
        if count == 0:
            unnumbered.append(name)
        for issue in issues:
            where = f"{name}:{issue.line} [{issue.stream}] after {issue.prev} got {issue.got}"
            if issue.kind == "gap":
                logger.warning(f"{where}, missing {issue.missing}")
            else:
                logger.info(f"{where} (out of order)")
        all_issues.extend(issues)

    with args.report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(Issue.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(vars(issue) for issue in all_issues)

    gaps = [i for i in all_issues if i.kind == "gap"]
    for name in unnumbered:
        logger.info(f"  no numbered items (not checked): {name}")
    logger.info(
        f"Done: {len(files)} files, {len({i.file for i in gaps})} with gaps, "
        f"{len(gaps)} gaps, {len(all_issues) - len(gaps)} out of order, "
        f"{len(unnumbered)} without numbering, {len(failed)} failed; report in {args.report}"
    )
    if gaps or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
