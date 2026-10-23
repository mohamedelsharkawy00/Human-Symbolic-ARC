"""Find copy-pasted code across the project, biggest collapse first.

How it works, in plain words: read every source file, slide a window of 6
meaningful lines (blank lines and comment-only lines don't count) down each
file, and fingerprint each window. Any fingerprint seen in two places means
those 6+ lines exist twice — a DRY cleanup candidate. Overlapping windows of
the same clone are chained into one maximal clone, so each duplicate block is
reported once with its true length. Ranked by lines saved if collapsed:
length x (copies - 1).

Shortcoming: it only sees textually identical lines. Two functions with the
same logic but different variable names score as a few small scraps or nothing,
so the biggest duplication in a file can be invisible here — reading the file
is the only way to catch structural twins.

Usage:  python scripts/find_duplicates.py [window_size]
Output: one line per clone group: savings, length, locations, then a preview.
"""

import collections
import os
import sys

ROOTS = ["include", "src"]
EXCLUDE_DIRS = {"third_party"}
EXTENSIONS = (".hpp", ".h", ".cpp")
WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def source_files():
    for root in ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for name in filenames:
                if name.endswith(EXTENSIONS):
                    yield os.path.join(dirpath, name)


def meaningful_lines(path):
    """(original line number, stripped text) for every non-blank, non-comment line."""
    in_block = False
    with open(path, encoding="utf-8", errors="ignore") as f:
        for number, line in enumerate(f, 1):
            text = line.strip()
            # Track /* ... */ blocks; drop lines that are comment from end to end.
            if in_block:
                if "*/" in text:
                    in_block = False
                    text = text.split("*/", 1)[1].strip()
                else:
                    continue
            if text.startswith("/*"):
                if "*/" in text:
                    text = text.split("*/", 1)[1].strip()
                else:
                    in_block = True
                    continue
            if text and not text.startswith("//"):
                yield number, text


def main():
    # Per file: the list of meaningful (line number, text) pairs.
    files = {path: list(meaningful_lines(path)) for path in source_files()}

    # fingerprint -> sorted tuple of (file, index into that file's meaningful lines)
    windows = collections.defaultdict(set)
    for path, lines in files.items():
        for k in range(len(lines) - WINDOW + 1):
            chunk = "\n".join(text for _, text in lines[k : k + WINDOW])
            windows[chunk].add((path, k))

    # Duplicate groups keyed by their exact spot set, so a clone one line longer
    # is simply the same spot set shifted by one — a chain we can follow.
    groups = {tuple(sorted(s)): 1 for s in windows.values() if len(s) >= 2}

    # A group is a chain start when the spot set shifted back by one is not
    # itself a duplicate group; follow forward shifts to find the clone length.
    clones = []
    for spots in groups:
        if tuple((f, k - 1) for f, k in spots) in groups:
            continue
        length = WINDOW
        step = spots
        while (step := tuple((f, k + 1) for f, k in step)) in groups:
            length += 1
        saved = length * (len(spots) - 1)
        clones.append((saved, length, spots))

    clones.sort(key=lambda c: (-c[0], c[2]))
    for saved, length, spots in clones:
        places = "  ".join(f"{f}:{files[f][k][0]}" for f, k in spots)
        print(f"saves {saved:4}  len {length:3}  {len(spots)}x  {places}")
        first_file, first_k = spots[0]
        print(f"     | {files[first_file][first_k][1]}")


if __name__ == "__main__":
    main()
