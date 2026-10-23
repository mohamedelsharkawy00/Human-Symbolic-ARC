# Where the elections stand

Parked 2026-08-28, mid-work. Everything below is UNCOMMITTED in the tree.

## What changed today, in order

1. **The fitting's budget counts cells touched, not commitments.** It allowed
   `(pieces + 1) x (cells + 1)` commitments, but choosing one rescans every
   required cell against every option against that option's cells — so a step
   was millions and the budget bounded nothing. 13f06aa5 ground a core
   indefinitely; now under a second. COMMITTED as ff08364.

2. **Hanging off the edge needs twins.** A piece may only sit half outside the
   grid if identical copies exist on EITHER grid — its own or the one it lands
   on. Both are read by `find_copies` before anything is matched, so no match
   vouches for itself. "Either grid" matters: 045e512c's key shape is one piece
   on the input and six copies on the output.

3. **Left over came apart into three.** CARRIED (an edge lands on the cell),
   MADE OR GONE (nothing over there is this cell's piece — made where it stands
   read from the output, went nowhere read from the input), UNEXPLAINED (its
   partner IS known and still cannot say where these cells came from). Only the
   third is the reading failing.

4. **The wholesale credit is gone.** A piece used to be credited with its whole
   footprint the moment ANYTHING spoke for it, so one edge on a 200-cell object
   accounted for 200 cells. That is what let junk readings score a perfect zero.

5. **But a piece that went somewhere WHOLE takes its cells with it.** Nothing
   ever lands back on the side a piece leaves from, so a fully matched piece had
   its own cells read as unaccounted — 00576224's input is one 2x2 block matched
   to all nine blocks of the output and its four cells still counted against it.
   Whole means the edge carries the piece's own shape with none of it off the
   grid, which keeps it clear of the leak in 4.

6. **The price no longer charges for MADE OR GONE.** Only unexplained.

7. **The pane counts from zero**, like the log and like the lists. READING and
   the four track steppers. The puzzle counter in the header is still 1-based.

## The elections now, against what the user says

    00576224   r66  MOTIF-SIZED [TURN]              0 left over   CORRECT
    13f06aa5   r16  ANY COLORS [AS IS]              0             correct
    03560426   r17  ANY COLORS [MOVE]               0             correct
    009d5c81   r4   ALL ONE COLOR [COLOR]           28 made/gone  correct
    025d127b   r0   ALL ONE COLOR [AS IS]           72 unexpl.    correct
    00dbd492   r32  EVERY CELL OF ONE COLOR [AS IS] 120 made/gone "okay for now, kinda too coarse"
    0692e18c   r9   ALL ONE COLOR [MOVE, HANG OFF]  55            wrong, parked separately
    05a7bcf2   r23  ANY COLORS [MOVE,TURN,COLOR]    190 made/gone WRONG
    045e512c   r3   ALL ONE COLOR [MOVE, TURN]      4             WRONG, new today

## The two open ones, in the user's words

**05a7bcf2** — "r15 is super wrong, there is no turning needed this puzzle, but
i know why, it is cuz we still didnt account for the weeping yet". TURN is
buying coverage that a proper account of the weeping would explain. The target
is r5, `ALL ONE COLOR [MOVE, COLOR]` — the reading whose fifteen lines on pair 1
were confirmed by hand as exactly right: eleven pieces that stayed put and four
reds that fell.

**045e512c** — "the new one is just wrong, it is also motif shenanigans, maybe
the weeping is that as well". It left r13 `[MOVE, COLOR, HANG OFF]` for r3
`[MOVE, TURN]`, 71 matches against 31. The suspicion is that both this and the
weeping are the same fault in how a motif is read.

## Standing preferences stated by the user

- Fewer freedoms is preferred, "in general", and it slightly outranks settling
  whether MADE OR GONE should be free or merely cheaper.
- Numbers are read off the pane, and the pane is now zero-based.

## What has NOT been done

- The pane still steps one LEFT OVER list; splitting it into MADE OR GONE and
  UNEXPLAINED was approved and not built.
- Nothing charges for freedoms. A reading pays for its cut and for what it
  cannot explain, and nothing for what it had to assume.
- Nine puzzles, read by hand. Nothing has been run across the 1120, and there is
  no regression check for elections at all.

## See also

- `A Piece Is Credited With Cells It Never Carried.md` — 0692e18c, and the leak
  that item 4 above closes.
- `Infer The Program Instead Of Searching For It.md` — the larger direction.

## Details worth knowing before touching any of this

**Inference is LAZY, and it will fool a measurement.**
`solver_executor.hpp` only re-reads a puzzle when the inference pane is open OR
the solver is unpaused:

    if (m_puzzle_loaded && !m_puzzle.inferred &&
        (inference_view != INFERENCE_OFF || !is_paused))

With the pane shut and the solver paused, jumping by `puzzle <id>` over the
command port switches what is displayed and analyses NOTHING — the log on disk
stays whatever it was. Several checks earlier this week were read off stale
logs before this was found. The reliable way is to set `settings/settings.txt`
to the puzzle's index and launch: the startup path always infers. Watch for the
inference log's mtime rather than the port's reply, which returns long before
the work is done.

**Reading numbers moved twice this week.** `MADE OF ONE CELL` was deleted, so
there are eighty readings and not ninety-six; and the pane now counts from zero.
Any number written down before 2026-08-26 wants checking against both.

**`twins_over_there` is pieces x landings**, a scan per piece with no index. It
is marked with a comment. Fine on everything seen so far, unmeasured on a grid
with hundreds of pieces.

**The per-reading refusal dump is gone** (commit ff08364). It is what found the
missing red piece of 05a7bcf2 pair 0, and it cost 1.7 MB of log per puzzle. If a
question needs it again it is in git history — `report_pieces_seen` in 5_log.hpp.

**The election's tie margin is `same_price = 0.001`**, and ties fall through to
fewest pieces used, then box cells. With MADE OR GONE now free, ties at zero
unexplained are common, so this margin decides more than it used to.

**The GUI's puzzle counter is still one-based** (`0012/1120`) while the command
port reports `idx=11`. The readings were unified; this was not.

**Session state left behind:** `settings/settings.txt` has `puzzle 11`,
`inference_view 2`, `paused 1`. The pane was turned on during measurement; the
puzzle index was moved around a lot and means nothing.
