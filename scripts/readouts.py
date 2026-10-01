"""Reads what scripts\\probe.bat writes (out\\probe\\<label>\\logs\\inference) beside
what arc_said printed (its stdout saved to a file: id, SOLVED or WRONG, program).

  python scripts/readouts.py explained <before-label> <after-label> <before-said> <after-said>
      The puzzles whose elected reading explains every cell -- nothing left
      over and no word missing -- yet are not solved, before and after a
      change. A new one is a reading that claims more than its program draws.

  python scripts/readouts.py missing <label> <said>
      Across the unsolved puzzles, each word the elected reading names as
      missing: how many puzzles name it, and how many lack only that word.

  python scripts/readouts.py solved <before-said> <after-said>
      The puzzles solved before and not after, and the other way round.
"""
import os
import re
import sys
from collections import Counter


def readouts(label):
    folder = os.path.join('out', 'probe', label, 'logs', 'inference')
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name), encoding='utf-8', errors='replace') as f:
            yield name[:-4], f.read()


def elected_block(readout):
    """The elected reading's lines under [fired], or None."""
    fired = re.search(r'^\[fired\].*?\n(.*?)^\[', readout, re.M | re.S)
    if not fired:
        return None
    block = re.search(r'^  reading \d+ .*ELECTED\n((?:    .*\n)*)', fired.group(1), re.M)
    return block.group(1) if block else None


def said(path, verdict='SOLVED'):
    with open(path, encoding='utf-8-sig') as f:
        return {line.split('\t')[0] for line in f if '\t%s\t' % verdict in line}


def explained(label):
    found = set()
    for pid, readout in readouts(label):
        block = elected_block(readout)
        if block and 'No Word' not in block and re.search(r'left over  0 of', block):
            found.add(pid)
    return found


def main(argv):
    if len(argv) == 5 and argv[0] == 'explained':
        before = explained(argv[1]) - said(argv[3])
        after = explained(argv[2]) - said(argv[4])
        print('explained but unsolved: before %d, after %d' % (len(before), len(after)))
        print('  new:  ', sorted(after - before))
        print('  gone: ', sorted(before - after))
    elif len(argv) == 3 and argv[0] == 'missing':
        solved = said(argv[2])
        named, only = Counter(), Counter()
        for pid, readout in readouts(argv[1]):
            if pid in solved:
                continue
            block = elected_block(readout)
            if not block:
                continue
            words = set(re.findall(r'No Word .*?no word for (.*)', block))
            named.update(words)
            if len(words) == 1:
                only.update(words)
        print('%-45s %8s %8s' % ('missing word', 'named', 'only it'))
        for word, count in named.most_common():
            print('%-45s %8d %8d' % (word, count, only[word]))
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
