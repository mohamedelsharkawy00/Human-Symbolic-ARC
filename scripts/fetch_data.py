"""Rebuild the local ARC data tree from the public ARC-AGI-2 repository.

Why this exists: the competition data may not be redistributed, so the repo
ships no puzzles. Anyone who clones (or any new git worktree) runs this once and
gets a complete local tree, in exactly the layout Kaggle mounts, so the same
relative paths work in both places.

Source : https://github.com/arcprize/ARC-AGI-2 (public, one task per file)
Output : kaggle/input/competitions/arc-prize-2026-arc-agi-2/
           arc-agi_training_challenges.json    1000 tasks
           arc-agi_training_solutions.json     1000 tasks
           arc-agi_evaluation_challenges.json   120 tasks
           arc-agi_evaluation_solutions.json    120 tasks
           arc-agi_test_challenges.json         240 tasks, no answers
           sample_submission.json               240 tasks

The public files carry test answers; a challenges file must not, so the answers
are split out into the matching solutions file -- the shape the loader expects
(load_challenges_json / load_solutions_json in include/core/15_puzzle_loader.hpp).

The test file is a STAND-IN: Kaggle swaps in the real hidden 240 at scoring
time. Locally it is the first 240 training ids with their answers removed,
which is what the competition ships as a placeholder.

Verified against the files Kaggle shipped: training, test and sample_submission
come out byte-identical. Six EVALUATION tasks differ -- 4a21e3da, abc82100,
b6f77b65, d8e07eb2, f560132c, faa9f03d -- same ids, but the public versions
carry one fewer test pair. The repo changelog (2025-04-14, 2025-04-17) records
those edits as fixes for off-by-one-pixel errors and ambiguous test pairs, so
the public copy is the CORRECTED one and the better proxy for the hidden set.
Neither copy is what actually scores a submission; the real test set is hidden.

Usage:
    python scripts/fetch_data.py            # write into kaggle/input/...
    python scripts/fetch_data.py --out DIR  # write somewhere else (for checking)
"""
import argparse, io, json, os, sys, urllib.request, zipfile

ARCHIVE = "https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip"
# Latest commit that touched the puzzles, so a README edit never reads as new data.
LATEST_DATA_COMMIT = ("https://api.github.com/repos/arcprize/ARC-AGI-2/commits"
                      "?path=data&per_page=1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "kaggle", "input", "competitions",
                           "arc-prize-2026-arc-agi-2")
# Records which upstream commit the local puzzles came from, so --check can tell
# "we are behind" from "we are current" without downloading anything.
STAMP_NAME = ".fetched_from.json"

# How many training tasks stand in for the hidden test set, and the grid Kaggle
# uses as a placeholder answer in the sample submission.
TEST_STAND_IN = 240
PLACEHOLDER_GRID = [[0, 0], [0, 0]]


def upstream_commit():
    """Return (sha, date) of the newest commit touching the puzzle data."""
    request = urllib.request.Request(LATEST_DATA_COMMIT,
                                     headers={"User-Agent": "arc-fetch-data"})
    with urllib.request.urlopen(request, timeout=15) as response:
        newest = json.load(response)[0]
    return newest["sha"], newest["commit"]["author"]["date"]


def read_stamp(directory):
    try:
        with open(os.path.join(directory, STAMP_NAME)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def check(directory):
    """Say whether upstream moved since we fetched. Never changes anything."""
    stamp = read_stamp(directory)
    if stamp is None:
        print("[*] No fetch stamp; run scripts/fetch_data.py to record one.")
        return
    try:
        sha, date = upstream_commit()
    except Exception:
        # Offline or rate-limited is not news. Stay quiet so builds are not noisy.
        return
    if sha == stamp.get("sha"):
        return
    print(f"[!] Puzzle data upstream changed on {date[:10]} "
          f"(local copy is from {stamp.get('date', '?')[:10]}).")
    print("[!] Run: python scripts/fetch_data.py   -- deliberately, since")
    print("[!] benchmark numbers before and after are not comparable.")


def download_tasks():
    """Return {"training": {id: task}, "evaluation": {id: task}} from the zip."""
    print(f"[*] Downloading {ARCHIVE}")
    with urllib.request.urlopen(ARCHIVE) as response:
        downloaded = response.read()
    print(f"[*] Got {len(downloaded) / 1024 / 1024:.1f} MB, reading tasks")

    splits = {"training": {}, "evaluation": {}}
    with zipfile.ZipFile(io.BytesIO(downloaded)) as archive:
        for entry in archive.namelist():
            parts = entry.split("/")
            # <repo>-main/data/<split>/<task_id>.json
            if len(parts) != 4 or parts[1] != "data" or not entry.endswith(".json"):
                continue
            split = parts[2]
            if split in splits:
                splits[split][parts[3][:-5]] = json.loads(archive.read(entry))
    for split, tasks in splits.items():
        print(f"[*] {split}: {len(tasks)} tasks")
    return splits


def split_answers(tasks):
    """Separate each task into a challenge (no test answers) and its answers."""
    challenges, solutions = {}, {}
    for task_id, task in sorted(tasks.items()):
        challenges[task_id] = {
            "train": task["train"],
            "test": [{"input": pair["input"]} for pair in task["test"]],
        }
        solutions[task_id] = [pair["output"] for pair in task["test"]]
    return challenges, solutions


def write_json(directory, name, payload):
    path = os.path.join(directory, name)
    with open(path, "w", newline="\n") as handle:
        json.dump(payload, handle)
    print(f"[*] Wrote {name} ({len(payload)} tasks, {os.path.getsize(path) / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true",
                        help="report whether upstream moved; change nothing")
    args = parser.parse_args()

    if args.check:
        check(args.out)
        return

    os.makedirs(args.out, exist_ok=True)
    splits = download_tasks()
    if not splits["training"] or not splits["evaluation"]:
        sys.exit("error: the archive held no task files; did the layout change?")

    for split, prefix in (("training", "training"), ("evaluation", "evaluation")):
        challenges, solutions = split_answers(splits[split])
        write_json(args.out, f"arc-agi_{prefix}_challenges.json", challenges)
        write_json(args.out, f"arc-agi_{prefix}_solutions.json", solutions)

    # Stand-in test set: the first training ids, answers withheld the way the
    # real hidden set withholds them.
    training_challenges, _ = split_answers(splits["training"])
    stand_in_ids = sorted(training_challenges)[:TEST_STAND_IN]
    write_json(args.out, "arc-agi_test_challenges.json",
               {task_id: training_challenges[task_id] for task_id in stand_in_ids})
    write_json(args.out, "sample_submission.json",
               {task_id: [{"attempt_1": PLACEHOLDER_GRID, "attempt_2": PLACEHOLDER_GRID}
                          for _ in training_challenges[task_id]["test"]]
                for task_id in stand_in_ids})

    # Record where this came from, so --check can compare without downloading.
    try:
        sha, date = upstream_commit()
        with open(os.path.join(args.out, STAMP_NAME), "w", newline="\n") as handle:
            json.dump({"sha": sha, "date": date}, handle, indent=2)
        print(f"[*] Fetched from commit {sha[:7]} ({date[:10]})")
    except Exception as problem:
        print(f"[!] Could not record the source commit: {problem}")

    print(f"[*] Done -> {args.out}")


if __name__ == "__main__":
    main()
