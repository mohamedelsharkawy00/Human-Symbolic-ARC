"""Generate small movement fixtures; this does not run the C++ solver."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "puzzles"
SHAPE = [(0, 0), (1, 0), (1, 1), (1, 2)]


def scene(height, width, top, left, shape=SHAPE):
    cells = [[0] * width for _ in range(height)]
    for row, col in shape:
        cells[top + row][left + col] = 2
    return cells


def example(kind, offset):
    top, left = 1, 1 + offset
    before = scene(6, 9, top, left)
    if kind == "movement_resize_stationary":
        after = scene(8, 9, top, left)
    elif kind == "movement_bidirectional":
        after = scene(6, 9, top, left + 2)
    else:
        # A clockwise turn maps (row, col) to (col, height - 1 - row).
        turned = [(col, 1 - row) for row, col in SHAPE]
        after = scene(6, 9, top, left, turned)
    return {"input": before, "output": after}


if __name__ == "__main__":
    for kind in ("movement_resize_stationary", "movement_bidirectional",
                 "movement_rotation_control"):
        fixture = {"train": [example(kind, 0), example(kind, 1)],
                   "test": [example(kind, 2)]}
        for pair in fixture["train"] + fixture["test"]:
            for grid in pair.values():
                assert 1 <= len(grid) <= 30
                assert all(len(row) == len(grid[0]) for row in grid)
                assert all(0 <= cell <= 9 for row in grid for cell in row)
            points = [set((r, c) for r, row in enumerate(pair[side])
                          for c, value in enumerate(row) if value == 2)
                      for side in ("input", "output")]
            if kind == "movement_resize_stationary":
                assert points[0] == points[1]
            elif kind == "movement_bidirectional":
                assert {(r, c + 2) for r, c in points[0]} == points[1]
            else:
                assert points[0] != points[1]
                assert len(points[0]) == len(points[1]) == 4
        (ROOT / (kind + ".json")).write_text(json.dumps(fixture, indent=2) + "\n")
        print(kind + ": fixture geometry checked")
