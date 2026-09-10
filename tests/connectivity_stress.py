"""Generate 30x30 ARC fixtures and check the embedding flood's contour votes."""
import json
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUZZLES = ROOT / "puzzles"
FOUR = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]


def scene(sealed=False, offset=0):
    grid = [[0] * 30 for _ in range(30)]
    walls = []
    for top in (1 + offset, 16 + offset):
        for left in (1 + offset, 16 + offset):
            wall = set()
            for r in range(12):
                for c in range(12):
                    edge = r in (0, 11) or c in (0, 11)
                    corner = r in (0, 11) and c in (0, 11)
                    tooth = 1 <= r <= 8 and c in (3, 5, 7, 9)
                    color = 2 if (edge and (sealed or not corner)) or tooth else 1
                    if corner and not sealed:
                        color = 0
                    grid[top + r][left + c] = color
                    if color == 2:
                        wall.add((top + r, left + c))
            walls.append(wall)
    return grid, walls


def contour(grid, wall, directions):
    outside = {(r, c) for r in range(30) for c in range(30)
               if (r in (0, 29) or c in (0, 29)) and (r, c) not in wall}
    queue = deque(outside)
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            p = r + dr, c + dc
            if 0 <= p[0] < 30 and 0 <= p[1] < 30 and p not in outside and p not in wall:
                outside.add(p)
                queue.append(p)
    votes = Counter(grid[r + dr][c + dc] for r, c in wall for dr, dc in FOUR
                    if (r + dr, c + dc) in outside)
    leaked = sum(grid[r][c] == 1 and (r, c) in outside
                 for r in range(min(r for r, _ in wall), max(r for r, _ in wall) + 1)
                 for c in range(min(c for _, c in wall), max(c for _, c in wall) + 1))
    return dict(votes), leaked


def pair(sealed=False, offset=0):
    grid, _ = scene(sealed, offset)
    return {"input": grid, "output": [[4 if c == 1 else c for c in row] for row in grid]}


if __name__ == "__main__":
    PUZZLES.mkdir(exist_ok=True)
    for sealed, name in ((False, "connectivity_stress_30"), (True, "connectivity_control_30")):
        fixture = {"train": [pair(sealed)], "test": [pair(sealed, 1)]}
        for example in fixture["train"] + fixture["test"]:
            for grid in example.values():
                assert len(grid) == 30 and all(len(row) == 30 for row in grid)
                assert all(0 <= c <= 9 for row in grid for c in row)
        (PUZZLES / (name + ".json")).write_text(json.dumps(fixture, indent=2) + "\n")
        grid, walls = scene(sealed)
        for wall in walls:
            four, enclosed4 = contour(grid, wall, FOUR)
            eight, leaked8 = contour(grid, wall, EIGHT)
            assert enclosed4 == 0
            if sealed:
                assert four == eight and leaked8 == 0
            else:
                assert leaked8 == 68
                assert four.get(1, 0) == 0 and eight[1] > eight[0]
        print(name, "first wall: 4-way", four, "8-way", eight,
              "blue cells reached by 8-way:", leaked8)
