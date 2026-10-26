"""Reads what arc_said writes (scripts\\said.bat [--as <label>]; unnamed is x64-Release): out\\<label>\\said.txt
(id, SOLVED or WRONG, program) and out\\<label>\\logs\\inference\\<id>.txt.

  python scripts/readouts.py explained <before> <after>
      The puzzles whose elected reading explains every cell -- nothing left
      over -- while their program does not fit the training pairs, before
      and after a change. A new one is a reading that claims more than its
      program draws. A program that fits every pair draws what it claims, so
      a wrong answer is not one.

  python scripts/readouts.py missing <label>
      Across the unsolved puzzles, each change the elected reading has no rule
      for: how many puzzles name it, and how many lack only that rule.

  python scripts/readouts.py solved <before> <after>
      The puzzles solved before and not after, and the other way round.
"""
import os
import re
import sys
from collections import Counter


def readouts(label):
    folder = os.path.join('out', label, 'logs', 'inference')
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name), encoding='utf-8', errors='replace') as f:
            yield name[:-4], f.read()


def elected_block(readout):
    """The elected reading's lines under [fired], or None. Where passes after
    the line read the rest on the picture it drew ([rest N]), the last pass
    kept: what it leaves is what the reading leaves. A pass saying no more than
    the one before it is only read, not said."""
    passes = re.findall(r'^\[rest \d+\][^\n]*?(?<!before it)\n(.*?)(?=^\[)', readout, re.M | re.S)
    if passes:
        block = re.search(r'^  reading \d+ .*ELECTED\n((?:    .*\n)*)', passes[-1], re.M)
        if block:
            return block.group(1)
    fired = re.search(r'^\[fired\].*?\n(.*?)^\[', readout, re.M | re.S)
    if not fired:
        return None
    block = re.search(r'^  reading \d+ .*ELECTED\n((?:    .*\n)*)', fired.group(1), re.M)
    return block.group(1) if block else None


def said(label, verdict='SOLVED'):
    path = os.path.join('out', label, 'said.txt')
    with open(path, encoding='utf-8-sig') as f:
        return {line.split('\t')[0] for line in f if '\t%s\t' % verdict in line}


def explained(label):
    found = set()
    for pid, readout in readouts(label):
        block = elected_block(readout)
        if block and re.search(r'left over  0 of', block):
            found.add(pid)
    return found


def main(argv):
    if len(argv) == 3 and argv[0] == 'explained':
        before = explained(argv[1]) - said(argv[1]) - said(argv[1], 'WRONG')
        after = explained(argv[2]) - said(argv[2]) - said(argv[2], 'WRONG')
        print('explained but unsolved: before %d, after %d' % (len(before), len(after)))
        print('  new:  ', sorted(after - before))
        print('  gone: ', sorted(before - after))
    elif len(argv) == 2 and argv[0] == 'missing':
        solved = said(argv[1])
        named, only = Counter(), Counter()
        for pid, readout in readouts(argv[1]):
            if pid in solved:
                continue
            block = elected_block(readout)
            if not block:
                continue
            changes = set(re.findall(r'No Rule .*?no rule for (.*)', block))
            named.update(changes)
            if len(changes) == 1:
                only.update(changes)
        print('%-45s %8s %8s' % ('no rule for', 'named', 'only it'))
        for change, count in named.most_common():
            print('%-45s %8d %8d' % (change, count, only[change]))
    elif len(argv) == 3 and argv[0] == 'solved':
        before, after = said(argv[1]), said(argv[2])
        print('lost:  ', sorted(before - after))
        print('gained:', sorted(after - before))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
