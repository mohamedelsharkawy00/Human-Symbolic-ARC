"""Split Kaggle's big ARC challenge files into per-task files matching the
local puzzles/ layout: { "train":[{input,output}], "test":[{input,output?}] }.

Input : kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_*_challenges.json
        kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_*_solutions.json
Output: puzzles_kaggle/{training,evaluation,test}/<task_id>.json

- train/eval: solutions merged back as test[i]["output"] (keeps eval harness working)
- test: no solutions exist (submit-only) -> test entries have input only
"""
import json, os, sys

SRC = os.path.join(os.path.dirname(__file__), "..", "kaggle", "input", "competitions", "arc-prize-2026-arc-agi-2")
DST = os.path.join(os.path.dirname(__file__), "..", "puzzles_kaggle")

SPLITS = [
    ("training",   "arc-agi_training_challenges.json",   "arc-agi_training_solutions.json"),
    ("evaluation", "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
    ("test",       "arc-agi_test_challenges.json",        None),
]

def main():
    for split, ch_file, sol_file in SPLITS:
        ch_path = os.path.join(SRC, ch_file)
        if not os.path.exists(ch_path):
            print(f"SKIP {split}: {ch_file} not found"); continue
        challenges = json.load(open(ch_path))
        solutions = {}
        if sol_file:
            sp = os.path.join(SRC, sol_file)
            if os.path.exists(sp):
                solutions = json.load(open(sp))
            else:
                print(f"WARN {split}: {sol_file} missing, outputs will be absent")

        out_dir = os.path.join(DST, split)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for tid, task in challenges.items():
            rec = {"train": task["train"], "test": []}
            sols = solutions.get(tid)  # list, one entry per test input, same order
            for i, t in enumerate(task["test"]):
                entry = {"input": t["input"]}
                if sols is not None and i < len(sols):
                    entry["output"] = sols[i]
                rec["test"].append(entry)
            with open(os.path.join(out_dir, f"{tid}.json"), "w") as f:
                json.dump(rec, f)
            n += 1
        print(f"{split}: wrote {n} task files -> puzzles_kaggle/{split}/")

if __name__ == "__main__":
    main()
