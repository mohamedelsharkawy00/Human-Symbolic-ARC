# Groups every puzzle by its dimension rule (Identity / Scale / Fixed /
# BoundingBox), mirroring derive_dimension_rule in include/inference/4_infer.hpp
# (incl. the cross-multiplied ratio check — no integer truncation).
# Usage: python scripts/list_puzzles_by_size_rule.py [rule_name]
#   no arg    -> summary counts + writes analysis/puzzles_by_size_rule.txt
#   rule_name -> prints just those ids (e.g. Identity), one per line
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "kaggle/input/competitions/arc-prize-2026-arc-agi-2"
SETS = {
    "train": "arc-agi_training_challenges.json",
    "eval":  "arc-agi_evaluation_challenges.json",
}


def dims(g):
    return (len(g[0]), len(g))  # (width, height)


def rule_for(train):
    iw0, ih0 = dims(train[0]["input"])
    ow0, oh0 = dims(train[0]["output"])
    gw, gh = math.gcd(ow0, iw0), math.gcd(oh0, ih0)
    wn, wd = ow0 // gw, iw0 // gw
    hn, hd = oh0 // gh, ih0 // gh

    fixed_ratio = all(
        dims(ex["output"])[0] * wd == dims(ex["input"])[0] * wn and
        dims(ex["output"])[1] * hd == dims(ex["input"])[1] * hn
        for ex in train)
    if fixed_ratio and (wn, wd, hn, hd) == (1, 1, 1, 1):
        return "Identity"
    if fixed_ratio:
        return "Scale"
    if all(dims(ex["output"]) == (ow0, oh0) for ex in train):
        return "Fixed"
    return "BoundingBox"


def main():
    groups = {}  # rule -> [(set, id)]
    for set_name, fname in SETS.items():
        tasks = json.loads((DATA / fname).read_text())
        for tid, task in sorted(tasks.items()):
            groups.setdefault(rule_for(task["train"]), []).append((set_name, tid))

    if len(sys.argv) > 1:
        want = sys.argv[1]
        for set_name, tid in groups.get(want, []):
            print(f"{set_name}  {tid}")
        return

    out = ROOT / "analysis" / "puzzles_by_size_rule.txt"
    out.parent.mkdir(exist_ok=True)
    lines = []
    total = sum(len(v) for v in groups.values())
    for rule in ("Identity", "Scale", "Fixed", "BoundingBox"):
        ids = groups.get(rule, [])
        n_train = sum(1 for s, _ in ids if s == "train")
        n_eval = len(ids) - n_train
        print(f"{rule:12} {len(ids):4}  (train {n_train}, eval {n_eval})")
        lines.append(f"== {rule} ({len(ids)}) ==")
        lines += [f"{s}  {t}" for s, t in ids]
        lines.append("")
    print(f"{'total':12} {total:4}")
    out.write_text("\n".join(lines))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
