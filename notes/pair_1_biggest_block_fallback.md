# Handoff: Pair 1 Missing from Biggest Block Reading (017c7c7b)

## Problem Statement
Puzzle 017c7c7b's pair 1 produces zero pieces in the `repeating_motif_big` (biggest block) reading, appearing only in pair 0 and pair 2. The reading summary shows "cut into 13" instead of "cut into 15", and pair 1 has no input or output pieces listed.

**User's requirement:** "If the smallest block found a motif, that should be the biggest for that pair, because the smallest one was able to find a motif."

## Root Cause
When `repeat_periods(biggest=true)` is applied to pair 1's input (3x6):
- **Pair 0:** period = 3x2 (found 3 copies vertically → tiled successfully)
- **Pair 1:** period = 3x6 (grid size — no divisor found that yields ≥2 copies)
- **Pair 2:** period = 3x3 (found 2 copies both ways → tiled successfully)

Pair 1's input has **no repeating block that tiles completely** (no period that divides evenly with ≥2 copies). So `repeat_periods(biggest=true)` returns the grid size itself.

When period = grid size:
- `segment_for_pairing` is called with step_w=3, step_h=6 on a 3x6 grid
- This creates exactly 1 block (the whole grid)
- That block should have ink (pair 1's input is color 1), so it should be added to `objs`
- But pair 1 produces **zero pieces**

## What Was Tried

### Attempt 1: Simple Fallback (lines 1846-1852)
```cpp
if (how == split_kind::repeating_motif_big && sw == in.width && sh == in.height) {
    repeat_periods(in, background, sw, sh, false);  // find smallest
}
```
**Result:** No change. Pair 1 still missing.

**Why it should work:** If biggest returns grid size (no tiling), call repeat_periods with biggest=false to find the smallest repeating pattern.

**Why it failed:** Unknown. The condition may not be evaluating to true, or the second call to repeat_periods also returns grid size for pair 1, or something else is filtering the pieces.

### Attempt 2: Always Find Both, Check Tiling (lines 1846-1858)
```cpp
int sw_big = sw, sh_big = sh, sw_small = sw, sh_small = sh;
repeat_periods(in, background, sw_big, sh_big, true);
repeat_periods(in, background, sw_small, sh_small, false);
if (how == split_kind::repeating_motif_big) {
    const bool biggest_tiled = (sw_big < in.width || sh_big < in.height);
    sw = biggest_tiled ? sw_big : sw_small;
    sh = biggest_tiled ? sh_big : sh_small;
} else {
    sw = sw_small;
    sh = sh_small;
}
```
**Result:** Still "cut into 13". Pair 1 still missing.

**Logic:** For biggest_motif, use biggest period only if it actually tiled (period < grid). Otherwise, use smallest.

**Why it should work:** Pair 1's condition `(3 < 3 || 6 < 6)` = false, so it should use sw_small. Then segment_for_pairing should create pieces from smallest period.

**Why it failed:** Either:
1. The code isn't compiled into the binary (unlikely, build succeeded)
2. `repeat_periods(false)` for pair 1 also returns 3x6 (pair 1 has no repeating pattern at all)
3. Pieces ARE created but then filtered out by `collect_edges` or elsewhere
4. Logic error in the condition

## Current Code State
**File:** `include/inference.hpp`, lines 1842-1858 (in `detect_object_pairing`)

Current code uses Attempt 2 logic (always find both periods). The build succeeds (exit code 0).

## What Needs Investigation

### 1. Does Pair 1 Have ANY Repeating Pattern?
Check if `repeat_periods(false)` for pair 1's input also returns 3x6, or if it finds a smaller period.
- **How:** Add a log line after both calls showing sw_small and sh_small values
- **Windows Defender issue:** Simple fprintf is OK; loops with `Start-Sleep` trigger alerts

### 2. Are Pieces Actually Being Created?
If smallest period is < grid size, segment_for_pairing SHOULD create pieces. Check if:
- The period is being calculated correctly
- `segment_for_pairing` is actually called (not skipped by an early return)
- Pieces are created but have zero ink and get filtered by line 1173

### 3. Are Pieces Filtered Later?
If pieces ARE created with smallest period, they might be filtered by:
- `collect_edges` (line 1857) — if edges contradict or fail other checks
- The election logic — if left_over is too high
- Something in the readout bridge

### 4. Is the Condition Logic Wrong?
The tiling check `(sw_big < in.width || sh_big < in.height)` may be incorrect.
- For pair 0: (3 < 3 || 2 < 6) = true → use biggest ✓
- For pair 1: (3 < 3 || 6 < 6) = false → use smallest ✓
- This looks correct, but maybe the intent is different

## Alternative Approaches

### Option A: Hardcode Pair 1 Test
Temporarily force pair 1 to use smallest period (3x4 from earlier observations) and see if pieces are created. This isolates whether the issue is in period-finding or piece-creation.

### Option B: Modify `repeat_periods` Return Value
Add a separate output to `repeat_periods` that says how many copies were found (0, 1, or 2+). Use that to decide fallback instead of comparing to grid size.

### Option C: Accept Pair 1 Abstention
If pair 1's input truly has no repeating pattern (both smallest and biggest return grid size), then abstaining from the biggest block reading is correct behavior. Verify this is the case, then document it.

### Option D: Change the Definition
Instead of "biggest block that tiles", use "largest block found" (even if it doesn't tile). This would make biggest = largest period regardless of copy count. Requires changing `repeat_periods` logic with biggest=true.

## Verification Checklist
Once a fix is attempted:
- [ ] Rebuild with `scripts\build.bat x64-Release NOLAUNCH`
- [ ] Load puzzle 017c7c7b in app
- [ ] Check `logs/inference/017c7c7b.txt` reading 1 summary
  - Should say "cut into 15" or higher (not 13)
  - Should include pair 1 pieces
- [ ] Check that pair 0 and pair 2 readings haven't changed
- [ ] Check other reference puzzles (025d127b, 03560426, etc.) are unaffected

## Files Involved
- **inference.hpp:1816-1867** — `detect_object_pairing()` function (period-finding and segmentation)
- **inference.hpp:1153-1182** — `cut_into_pieces()` (repeating_motif block creation)
- **core_types.hpp:54-92** — `repeat_periods()` (single home of period algorithm)
- **inference.hpp:1375-1460** — `collect_edges()` (correspondence matching)

## Log Evidence
```
[pair 0] period for smallest: 2x2 (input 3x6)
[pair 0] period for biggest: 3x2 (input 3x6)

[pair 1] period for smallest: 3x4 (input 3x6)
[pair 1] period for biggest: 3x6 (input 3x6)  ← grid size, no tiling

[pair 2] period for smallest: 2x3 (input 3x6)
[pair 2] period for biggest: 3x3 (input 3x6)
```

Reading 1 result (current):
```
reading 1  cut into 13, used 7 (7 exact, 0 recolored, 0 sits on), 14 left over
  pair 0 pieces: 3 input + 4 output = 7 total ✓
  pair 1 pieces: 0 input + 0 output = 0 total ✗
  pair 2 pieces: 2 input + 3 output = 5 total ✓
  Total: 7 + 0 + 5 = 12 pieces cut (actually 13 listed?)
```

Expected (if fallback works):
```
reading 1  cut into 15, used 7 (7 exact, 0 recolored, 0 sits on), 14 left over
  pair 0 pieces: 3 input + 4 output = 7 total ✓
  pair 1 pieces: X input + Y output > 0 total
  pair 2 pieces: 2 input + 3 output = 5 total ✓
```

## Notes
- Windows Defender flags PowerShell loops with `Start-Sleep` as `Trojan:Win32/ClickFix.CCJIMTB` — avoid debugging with loops
- The user's standing constraint: "don't decide stuff without me just say results" — once this is working, show the result before deciding which reading to elect
- User preference: "tell the truth first" — if pair 1 genuinely has no repeating pattern, say so instead of faking it
