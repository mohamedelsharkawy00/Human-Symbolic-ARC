// This file defines what a program's ops hand each other.
// Operators are combined into sentences, and those sentences together
// form a program.
//
// The earlier chapters describe what happens in the training examples.
// To solve a puzzle, we need a program that can make those changes.
// Its operators can find objects, move them, change their colors,
// and draw the result.
//
// 14_vocabulary.hpp defines the operators; src/ops.cpp runs them
// (environment::draw), on the objects defined here.
#pragma once
// MAX_OBJECTS_IN_A_LIST, and the vocabulary.
#include "7_facts.hpp"
#include "14_vocabulary.hpp"
#include <cstdint>

namespace arc {

// One object a program holds: the frame it sits in on the grid, and its cells
// in that frame.
//
// `selected` is the SELECT slot at object depth: one flag per object, not per
// cell. Selection works like Excel: nothing is selected at birth, selectors
// ADD whole objects, InvertSelection flips membership, SelectAll marks all.
// Acts touch ONLY selected objects — no selection means acts do nothing, never
// "everything". Sub-object reach comes from carving the piece into its own
// object first (Split/Colors), then selecting it — not from marking cells.
struct held_object {
    int16_t x0 = 0, y0 = 0;
    int16_t x1 = 0, y1 = 0;
    int16_t width = 0, height = 0;
    bool selected = false;
    cells mask;
};

// The list an op reads and the list it hands on: as long as a grid has cells,
// so a list never drops an object.
struct object_list {
    held_object objects[MAX_OBJECTS_IN_A_LIST];
    int count = 0;
};

}  // namespace arc
