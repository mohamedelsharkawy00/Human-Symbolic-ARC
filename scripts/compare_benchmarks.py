#!/usr/bin/env python3
"""Compare ARC benchmark logs, including logs without summary headers."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FIELD_RE = re.compile(r"^(Puzzle|Train|Test|Elapsed|Attempts|Nodes|Best Score|Max Depth):\s*(.*)$")
SUMMARY_RE = re.compile(r"^(Puzzles Run|Train Solved|Test Solved|Bench Time|Avg APS):\s*(.*)$")
FILENAME_DATE_RE = re.compile(
    r"(?:full_)?benchmark_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_(AM|PM)\.txt$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Entry:
    program: str
    puzzle: str
    train: str
    test: str
    elapsed_text: str
    elapsed_seconds: Optional[float]
    attempts: Optional[int]
    nodes: Optional[int]
    best_score: Optional[float]
    max_depth: Optional[int]
    line_number: int

    @property
    def aps(self) -> Optional[float]:
        if self.elapsed_seconds is None or self.elapsed_seconds <= 0 or self.attempts is None:
            return None
        return self.attempts / self.elapsed_seconds


@dataclass
class Benchmark:
    path: Path
    date: Optional[datetime]
    entries: List[Entry]
    embedded_summary: Dict[str, str]

    @property
    def label(self) -> str:
        return self.path.name

    @property
    def has_summary(self) -> bool:
        return bool(self.embedded_summary)

    @property
    def records(self) -> int:
        return len(self.entries)

    @property
    def blank_ids(self) -> int:
        return sum(1 for entry in self.entries if not entry.puzzle)

    @property
    def unique_ids(self) -> int:
        return len({entry.puzzle for entry in self.entries if entry.puzzle})

    @property
    def duplicate_ids(self) -> int:
        counts = Counter(entry.puzzle for entry in self.entries if entry.puzzle)
        return sum(1 for count in counts.values() if count > 1)

    @property
    def train_solved(self) -> int:
        return sum(1 for entry in self.entries if solved(entry.train))

    @property
    def test_solved(self) -> int:
        return sum(1 for entry in self.entries if solved(entry.test))

    @property
    def timeouts(self) -> int:
        return sum(1 for entry in self.entries if entry.train.strip().lower() == "timeout")

    @property
    def total_elapsed(self) -> Optional[float]:
        vals = [entry.elapsed_seconds for entry in self.entries if entry.elapsed_seconds is not None]
        return sum(vals) if vals else None

    @property
    def avg_aps(self) -> Optional[float]:
        vals = [entry.aps for entry in self.entries if entry.aps is not None]
        return mean(vals) if vals else None

    def best_by_puzzle(self) -> Dict[str, Entry]:
        best: Dict[str, Entry] = {}
        for entry in self.entries:
            if not entry.puzzle:
                continue
            old = best.get(entry.puzzle)
            if old is None or entry_rank(entry) > entry_rank(old):
                best[entry.puzzle] = entry
        return best


def solved(value: str) -> bool:
    return value.strip().lower() == "solved"


def parse_duration(text: str) -> Optional[float]:
    value = text.strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return float(value)
    except ValueError:
        return None


def parse_int(text: str) -> Optional[int]:
    try:
        return int(text.strip())
    except ValueError:
        return None


def parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_date_from_name(path: Path) -> Optional[datetime]:
    match = FILENAME_DATE_RE.search(path.name)
    if not match:
        return None
    date, hour, minute, second, ampm = match.groups()
    return datetime.strptime(f"{date} {hour}:{minute}:{second} {ampm}", "%Y-%m-%d %I:%M:%S %p")


def parse_benchmark(path: Path) -> Benchmark:
    text = path.read_text(encoding="utf-8", errors="replace")
    embedded_summary: Dict[str, str] = {}
    entries: List[Entry] = []
    current: Optional[Dict[str, str]] = None
    program_candidate = ""

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        summary_match = SUMMARY_RE.match(line)
        if summary_match:
            embedded_summary[summary_match.group(1)] = summary_match.group(2).strip()
            continue

        field_match = FIELD_RE.match(line)
        if field_match:
            key = field_match.group(1)
            value = field_match.group(2).strip()
            if key == "Puzzle":
                if current is not None:
                    entries.append(entry_from_fields(current))
                current = {
                    "program": program_candidate,
                    "Puzzle": value,
                    "line_number": str(line_number),
                }
                program_candidate = ""
            elif current is not None:
                current[key] = value
                if key == "Max Depth":
                    entries.append(entry_from_fields(current))
                    current = None
                    program_candidate = ""
            continue

        if line.startswith("=") or "BENCHMARK SUMMARY" in line:
            continue
        program_candidate = line

    if current is not None:
        entries.append(entry_from_fields(current))

    return Benchmark(path=path, date=parse_date_from_name(path), entries=entries, embedded_summary=embedded_summary)


def entry_from_fields(fields: Dict[str, str]) -> Entry:
    elapsed_text = fields.get("Elapsed", "")
    return Entry(
        program=fields.get("program", ""),
        puzzle=fields.get("Puzzle", ""),
        train=fields.get("Train", ""),
        test=fields.get("Test", ""),
        elapsed_text=elapsed_text,
        elapsed_seconds=parse_duration(elapsed_text),
        attempts=parse_int(fields.get("Attempts", "")),
        nodes=parse_int(fields.get("Nodes", "")),
        best_score=parse_float(fields.get("Best Score", "")),
        max_depth=parse_int(fields.get("Max Depth", "")),
        line_number=parse_int(fields.get("line_number", "")) or 0,
    )


def entry_rank(entry: Entry) -> Tuple[int, int, float, float, float]:
    score = entry.best_score if entry.best_score is not None else float("-inf")
    elapsed = entry.elapsed_seconds if entry.elapsed_seconds is not None else float("inf")
    aps = entry.aps if entry.aps is not None else 0.0
    return (1 if solved(entry.test) else 0, 1 if solved(entry.train) else 0, score, -elapsed, aps)


def format_pct(count: int, total: int) -> str:
    if total <= 0:
        return "0 / 0 (0.0%)"
    return f"{count} / {total} ({100.0 * count / total:.1f}%)"


def format_number(value: Optional[float], decimals: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{decimals}f}"


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def status(entry: Entry) -> str:
    if solved(entry.test):
        return "test solved"
    if solved(entry.train):
        return f"train solved / test {entry.test or 'unknown'}"
    return entry.train or "unknown"


def compare_pair(old: Benchmark, new: Benchmark) -> Dict[str, object]:
    old_map = old.best_by_puzzle()
    new_map = new.best_by_puzzle()
    common = sorted(set(old_map) & set(new_map))
    train_gains = train_losses = test_gains = test_losses = 0
    elapsed_ratios: List[float] = []
    aps_ratios: List[float] = []

    for puzzle in common:
        old_entry = old_map[puzzle]
        new_entry = new_map[puzzle]
        old_train = solved(old_entry.train)
        new_train = solved(new_entry.train)
        old_test = solved(old_entry.test)
        new_test = solved(new_entry.test)

        train_gains += int(new_train and not old_train)
        train_losses += int(old_train and not new_train)
        test_gains += int(new_test and not old_test)
        test_losses += int(old_test and not new_test)

        if old_entry.elapsed_seconds and new_entry.elapsed_seconds and new_entry.elapsed_seconds > 0:
            elapsed_ratios.append(old_entry.elapsed_seconds / new_entry.elapsed_seconds)
        if old_entry.aps and new_entry.aps and old_entry.aps > 0:
            aps_ratios.append(new_entry.aps / old_entry.aps)

    return {
        "common": len(common),
        "train_gains": train_gains,
        "train_losses": train_losses,
        "test_gains": test_gains,
        "test_losses": test_losses,
        "median_elapsed_speedup": median(elapsed_ratios) if elapsed_ratios else None,
        "median_aps_ratio": median(aps_ratios) if aps_ratios else None,
    }


def latest_change_rows(old: Benchmark, new: Benchmark, max_changes: int) -> Tuple[List[Tuple[str, Entry, Entry]], List[Tuple[str, Entry, Entry]]]:
    old_map = old.best_by_puzzle()
    new_map = new.best_by_puzzle()
    test_gains: List[Tuple[str, Entry, Entry]] = []
    test_losses: List[Tuple[str, Entry, Entry]] = []
    for puzzle in sorted(set(old_map) & set(new_map)):
        old_entry = old_map[puzzle]
        new_entry = new_map[puzzle]
        if solved(new_entry.test) and not solved(old_entry.test):
            test_gains.append((puzzle, old_entry, new_entry))
        elif solved(old_entry.test) and not solved(new_entry.test):
            test_losses.append((puzzle, old_entry, new_entry))
    return test_gains[:max_changes], test_losses[:max_changes]


def benchmark_sort_key(benchmark: Benchmark) -> Tuple[datetime, str]:
    return (benchmark.date or datetime.fromtimestamp(benchmark.path.stat().st_mtime), benchmark.path.name)


def discover_paths(paths: Sequence[str]) -> List[Path]:
    if not paths:
        paths = ["benchmarks"]
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.glob("*.txt")))
        else:
            found.append(path)
    return sorted({path.resolve() for path in found})


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, benchmarks: Sequence[Benchmark], max_changes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        "# Benchmark Comparison",
        "",
        f"Generated: {generated}",
        "",
        "Summaries are synthesized from per-puzzle records when a log has no embedded summary. "
        "Per-puzzle comparisons use the best record for each nonblank puzzle ID in each file.",
        "",
        "## Overview",
        "",
    ]

    overview_rows = []
    for benchmark in benchmarks:
        source = "embedded" if benchmark.has_summary else "synthesized"
        if benchmark.embedded_summary.get("Puzzles Run"):
            source += f" ({benchmark.embedded_summary['Puzzles Run']} runs)"
        overview_rows.append(
            [
                benchmark.label,
                source,
                benchmark.records,
                benchmark.unique_ids,
                benchmark.blank_ids,
                benchmark.duplicate_ids,
                format_pct(benchmark.train_solved, benchmark.records),
                format_pct(benchmark.test_solved, benchmark.records),
                benchmark.timeouts,
                format_seconds(benchmark.total_elapsed),
                format_number(benchmark.avg_aps, 2),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Benchmark",
                "Summary",
                "Records",
                "Unique IDs",
                "Blank IDs",
                "Dup IDs",
                "Train Solved",
                "Test Solved",
                "Timeouts",
                "Total Elapsed",
                "Avg APS",
            ],
            overview_rows,
        )
    )

    if len(benchmarks) >= 2:
        lines.extend(["", "## Pairwise Deltas", ""])
        pair_rows = []
        for old, new in zip(benchmarks, benchmarks[1:]):
            diff = compare_pair(old, new)
            pair_rows.append(
                [
                    f"{old.label} -> {new.label}",
                    diff["common"],
                    f"+{diff['train_gains']} / -{diff['train_losses']}",
                    f"+{diff['test_gains']} / -{diff['test_losses']}",
                    format_ratio(diff["median_elapsed_speedup"]),
                    format_ratio(diff["median_aps_ratio"]),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "Comparison",
                    "Common IDs",
                    "Train +/-",
                    "Test +/-",
                    "Median Elapsed Speedup",
                    "Median APS Ratio",
                ],
                pair_rows,
            )
        )

        latest = benchmarks[-1]
        lines.extend(["", "## Latest Vs Earlier", ""])
        latest_rows = []
        for old in benchmarks[:-1]:
            diff = compare_pair(old, latest)
            latest_rows.append(
                [
                    f"{old.label} -> {latest.label}",
                    diff["common"],
                    f"+{diff['train_gains']} / -{diff['train_losses']}",
                    f"+{diff['test_gains']} / -{diff['test_losses']}",
                    format_ratio(diff["median_elapsed_speedup"]),
                    format_ratio(diff["median_aps_ratio"]),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "Comparison",
                    "Common IDs",
                    "Train +/-",
                    "Test +/-",
                    "Median Elapsed Speedup",
                    "Median APS Ratio",
                ],
                latest_rows,
            )
        )

        lines.extend(["", "## Latest Test Changes", ""])
        for old in benchmarks[:-1]:
            gains, losses = latest_change_rows(old, latest, max_changes)
            lines.extend(["", f"### Vs {old.label}", ""])
            if gains:
                lines.extend(["Test-solved gains:", "", change_table(gains), ""])
            else:
                lines.append("No test-solved gains on shared puzzle IDs.")
                lines.append("")
            if losses:
                lines.extend(["Test-solved losses:", "", change_table(losses)])
            else:
                lines.append("No test-solved losses on shared puzzle IDs.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}x"


def change_table(rows: Sequence[Tuple[str, Entry, Entry]]) -> str:
    table_rows = []
    for puzzle, old_entry, new_entry in rows:
        table_rows.append(
            [
                puzzle,
                status(old_entry),
                status(new_entry),
                old_entry.elapsed_text or "n/a",
                new_entry.elapsed_text or "n/a",
                truncate(new_entry.program, 72),
            ]
        )
    return markdown_table(["Puzzle", "Previous", "Latest", "Prev Elapsed", "Latest Elapsed", "Latest Program"], table_rows)


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def write_overview_csv(path: Path, benchmarks: Sequence[Benchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "benchmark",
                "has_embedded_summary",
                "records",
                "unique_ids",
                "blank_ids",
                "duplicate_ids",
                "train_solved",
                "test_solved",
                "timeouts",
                "total_elapsed_seconds",
                "avg_aps",
            ]
        )
        for benchmark in benchmarks:
            writer.writerow(
                [
                    benchmark.label,
                    benchmark.has_summary,
                    benchmark.records,
                    benchmark.unique_ids,
                    benchmark.blank_ids,
                    benchmark.duplicate_ids,
                    benchmark.train_solved,
                    benchmark.test_solved,
                    benchmark.timeouts,
                    f"{benchmark.total_elapsed:.4f}" if benchmark.total_elapsed is not None else "",
                    f"{benchmark.avg_aps:.4f}" if benchmark.avg_aps is not None else "",
                ]
            )


def write_matrix_csv(path: Path, benchmarks: Sequence[Benchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    maps = [(benchmark, benchmark.best_by_puzzle()) for benchmark in benchmarks]
    puzzles = sorted(set().union(*(set(entry_map) for _, entry_map in maps)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["puzzle"]
        for benchmark, _ in maps:
            prefix = benchmark.label
            header.extend(
                [
                    f"{prefix}:train",
                    f"{prefix}:test",
                    f"{prefix}:elapsed_seconds",
                    f"{prefix}:attempts",
                    f"{prefix}:aps",
                    f"{prefix}:best_score",
                    f"{prefix}:program",
                ]
            )
        writer.writerow(header)
        for puzzle in puzzles:
            row = [puzzle]
            for _, entry_map in maps:
                entry = entry_map.get(puzzle)
                if entry is None:
                    row.extend(["", "", "", "", "", "", ""])
                else:
                    row.extend(
                        [
                            entry.train,
                            entry.test,
                            f"{entry.elapsed_seconds:.4f}" if entry.elapsed_seconds is not None else "",
                            entry.attempts if entry.attempts is not None else "",
                            f"{entry.aps:.4f}" if entry.aps is not None else "",
                            entry.best_score if entry.best_score is not None else "",
                            entry.program,
                        ]
                    )
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Benchmark files or directories. Defaults to ./benchmarks.")
    parser.add_argument("--out", default="benchmarks/benchmark_comparison.md", help="Markdown report path.")
    parser.add_argument("--csv", default="benchmarks/benchmark_comparison.csv", help="Overview CSV path.")
    parser.add_argument(
        "--matrix-csv",
        default="benchmarks/benchmark_puzzle_matrix.csv",
        help="Per-puzzle matrix CSV path.",
    )
    parser.add_argument("--max-changes", type=int, default=50, help="Max latest-vs-previous changes to list.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = discover_paths(args.paths)
    if not paths:
        raise SystemExit("No benchmark .txt files found.")

    benchmarks = [parse_benchmark(path) for path in paths]
    benchmarks.sort(key=benchmark_sort_key)

    write_markdown(Path(args.out), benchmarks, args.max_changes)
    write_overview_csv(Path(args.csv), benchmarks)
    write_matrix_csv(Path(args.matrix_csv), benchmarks)

    print(f"Compared {len(benchmarks)} benchmark files.")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.matrix_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
