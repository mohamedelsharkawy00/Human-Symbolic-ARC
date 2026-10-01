# Some objects are arrangements, and no cut makes one

Parked 2026-08-12. Raised on 11e1fe23, where every cut ties at roughly the same
mediocre reading and none of them is right.

## The puzzle

Three scattered dots in the input, nothing else:

    2 at row 3, col 8
    8 at row 9, col 2
    6 at row 9, col 8

The output keeps all three and adds a compact cluster around a new centre:

    5 at row 6, col 5      the centre, a colour not in the input
    2 at row 5, col 6      up and right of it
    8 at row 7, col 4      down and left
    6 at row 7, col 6      down and right

The cluster is a MINIATURE of the three dots. Relative to the 8, the 2 is up and
right and the 6 is right — and in the cluster, relative to the 8, the 2 is up
and right and the 6 is right. Same arrangement, shrunk to touching distance,
with a marker dropped at its centre.

## Why no cut helps

The dots are single cells with nothing beside them, so every cut produces the
same objects — one per dot — and they all score alike:

    left  5  edges 15   a group of cells that touch, all one color
    left  5  edges 15   made of one cell
    left  6  edges 14   every cell of one color, wherever it exists
    left  8  edges 12   a group of cells that touch, any colors

The three input dots match the three that stayed put, and the four new cells are
left over. That is the correct reading of what the cuts offer, and it says
nothing about the puzzle.

## What is missing

The object here is not a region of the picture. It is the ARRANGEMENT of three
separate marks — their relative directions, taken as one thing. Every cut we
have makes objects out of cells that are together: touching, or sharing a
colour, or falling in the same block. None can make an object out of marks that
are deliberately apart, whose whole content is how they sit relative to each
other.

So this is not a cut that is missing from the list of cuts. It is a second kind
of object: a constellation. The pieces of it are already objects; what has no
name is the set and its shape.

## Why it is not a small change

Everything downstream assumes an object is a run of cells with a bounding box.
An arrangement has no cells of its own — its box is the box of its members, its
"shape" is a set of offsets, and it can be scaled, which no object can today.
Correspondence would need to match a shape of offsets against another shape of
offsets, at a different size.

Worth deciding rather than drifting into: whether inference grows a second kind
of object, or whether arrangements are a channel of their own beside
correspondence, reading relations between objects rather than objects.
