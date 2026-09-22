// The words a program is written in, as 0_core.hpp holds the words a grid is.
// Numbered for where it is read, not for what it includes: the ops are named
// for what 1-6 find in a grid, and some lean on facts 7_infer.hpp elects. It
// compiles against 0_core.hpp alone, and the 7 is no licence to change that.
// Read by 8_sentences.hpp, which spells programs, and 11_dsl.cuh, which runs
// them, from the same definitions.
#pragma once
#include "0_core.hpp"
#include <string_view>

// What an op does to the layer, in pipeline-authoring order (a single solution
// line reads left-to-right through these): bring a source in (GET) -> cut it
// down (REDUCE) -> pick objects (SELECT) -> move it (TRANSLATE) -> recolor or
// reason (MAP) -> grow it (GENERATE) -> merge into the canvas (COMBINE).
// The generator's four sentence roles fold out of this: the middle four are ACT.
// Order is frozen — the GUI indexes its tab rows by these ordinals.
enum class op_role { get, reduce, select, translate, map, generate, combine, count };

// --- Op vocabulary: the single home of the DSL op list ---
// CPU (search, GUI, serializer) and GPU (kernel dispatch) both derive from
// this one macro, so the two sides cannot drift apart.
// Every op is unary (one Objects input, one Objects output); nodes form a chain.
// ROLE is a column here, not a list elsewhere, so a new op cannot be added
// without saying what it does — the macro will not compile without it.
// AN OP'S ARGUMENT IS A RANGE, LOW AND HIGH, AND THAT IS THE ONLY PLACE IT IS
// SAID. How many values is HIGH - LOW + 1, which value is LOW + index, and
// whether the op takes an argument at all is LOW != HIGH -- three facts read
// off two numbers, so none of them can disagree with the others, and no index
// is interpreted anywhere. An op that takes nothing writes 0, 0.
// THE ORDER IS THE SEARCH ORDER, simplest first: fewest variants, then fewest
// lines of code. Ties: turns before flips, rights before lefts, inputs before
// outputs, canvas as Union/Underlay/Intersection/Difference, then fewer letters.
#define NODE_PRIMITIVES /* Schema: X(ID, NAME, LOW, HIGH, ROLE) */               \
  X(SelectAll, "SelectAll", 0, 0, select)                                                                        \
  X(InvertSelection, "InvertSelection", 0, 0, select)                                                            \
  X(TurnRight, "TurnRight", 0, 0, translate)                                                                     \
  X(TurnLeft, "TurnLeft", 0, 0, translate)                                                                       \
  X(FlipHorizontal, "FlipHorizontal", 0, 0, translate)                                                           \
  X(FlipVertical, "FlipVertical", 0, 0, translate)                                                               \
  X(Input, "Input", 0, 0, get)                                                                                   \
  X(InputWithoutBackground, "InputWithoutBackground", 0, 0, get)                                                 \
  X(Output, "Output", 0, 0, get)                                                                                 \
  X(CurrentObjects, "CurrentObjects", 0, 0, get)                                                                 \
  X(Origin, "Origin", 0, 0, translate)                                                                           \
  X(Union, "Union", 0, 0, combine)                                                                               \
  /* DeleteSelection consumes the marks, it does not make them: an act. */       \
  X(DeleteSelection, "DeleteSelection", 0, 0, reduce)                                                            \
  /* The canvas-readers judge by what lies beneath, not by the object: acts. */  \
  X(ReadUnder, "ReadUnder", 0, 0, map)                                                                           \
  X(Outline, "Outline", 0, 0, reduce)                                                                            \
  X(SelectMostCommonColor, "SelectMostCommonColor", 0, 0, select)                                                \
  /* Colors extracts the palette as objects — a producing act, not a source. */  \
  X(Colors, "Colors", 0, 0, map)                                                                                 \
  X(Trim, "Trim", 0, 0, reduce)                                                                                  \
  X(OverlapCorner, "OverlapCorner", 0, 0, map)                                                                   \
  X(Underlay, "Underlay", 0, 0, combine)                                                                         \
  /* Withhold closes the beat without painting, so a sentence that only sets  */ \
  /* up the hand is sayable and does not read as damage to the canvas.        */ \
  X(Withhold, "Withhold", 0, 0, combine)                                                                         \
  X(Fit, "Fit", 0, 0, reduce)                                                                                    \
  X(Intersection, "Intersection", 0, 0, combine)                                                                 \
  X(Fractal, "Fractal", 0, 0, generate)                                                                          \
  X(Difference, "Difference", 0, 0, combine)                                                                     \
  X(UnionList, "UnionList", 0, 0, reduce)                                                                        \
  X(AlignToCanvas, "AlignToCanvas", 0, 0, translate)                                                             \
  X(IntersectionList, "IntersectionList", 0, 0, reduce)                                                          \
  X(SizeRankColorMap, "SizeRankColorMap", 0, 0, map)                                                             \
  X(Pattern, "Pattern", 0, 0, generate)                                                                          \
  X(ProjectOnSeeds, "ProjectOnSeeds", 0, 0, map)                                                                 \
  X(Bridge, "Bridge", 0, 0, generate)                                                                            \
  X(Partition, "Partition", 0, 0, generate)                                                                      \
  X(RestoreOrientation, "RestoreOrientation", 0, 0, translate)                                                   \
  X(ShapeColorMap, "ShapeColorMap", 0, 0, map)                                                                   \
  X(Seeds, "Seeds", 0, 0, get)                                                                                   \
  /* Partition cuts on the walls a selection marks; Panes cuts on the cut  */    \
  /* the split elected, which is the same act with a surer hand.           */        \
  X(Panes, "Panes", 0, 0, generate)                                                                              \
  X(VoteDenoise, "VoteDenoise", 0, 0, map)                                                                       \
  X(ShapeToInteriorColorMap, "ShapeToInteriorColorMap", 0, 0, map)                                               \
  X(SnapToAnchors, "SnapToAnchors", 0, 0, map)                                                                   \
  X(SortBySize, "SortBySize", 0, 1, map)                                                                         \
  X(DiagTile, "DiagTile", 0, 1, generate)                                                                        \
  X(NormalizeOrientation, "NormalizeOrientation", 0, 1, translate)                                               \
  /* A skew leans one way or the other, and 0 is the lean nobody takes. */       \
  X(Skew, "Skew", -1, 1, translate)                                                                              \
  X(Split, "Split", 0, 2, generate)                                                                              \
  /* Cells slide a step and pile at the wall, so a step past the wall says */    \
  X(SlamColumn, "SlamColumn", -1, 1, translate)                                                                  \
  /* nothing new: the answers run out at the object, not at the grid. */         \
  X(SlamRow, "SlamRow", -1, 1, translate)                                                                        \
  X(Tile, "Tile", 0, 3, generate)                                                                                \
  X(SelectObjectsWithHoles, "SelectObjectsWithHoles", 0, 4, select)                                              \
  X(Shift, "Shift", 0, 7, translate)                                                                             \
  X(MoveToEdge, "MoveToEdge", 0, 7, translate)                                                                   \
  X(ExtendTo, "ExtendTo", 0, 7, generate)                                                                        \
  X(ExtendDiagonal, "ExtendDiagonal", 2, 10, generate)                                                           \
  X(ExtendOrthogonal, "ExtendOrthogonal", 2, 10, generate)                                                       \
  X(ScaleUp, "ScaleUp", 2, 10, translate)                                                                        \
  X(ScaleDown, "ScaleDown", 2, 10, translate)                                                                    \
  X(Layout, "Layout", 2, 10, translate)                                                                          \
  X(SelectColor, "SelectColor", 0, arc::COLOR_COUNT - 1, select)                                                 \
  X(SelectObjectsContainingColor, "SelectObjectsContainingColor", 0, arc::COLOR_COUNT - 1, select)               \
  X(SelectLargestObject, "SelectLargestObject", 0, 9, select)                                                    \
  X(SelectSmallestObject, "SelectSmallestObject", 0, 9, select)                                                  \
  X(AddColor, "AddColor", 0, arc::COLOR_COUNT - 1, generate)                                                     \
  X(MaskBy, "MaskBy", 0, 10, map)                                                                                \
  X(MaskOut, "MaskOut", 0, 10, map)                                                                              \
  X(InteriorColor, "InteriorColor", -1, arc::COLOR_COUNT - 1, map)                                               \
  X(Recolor, "Recolor", 0, 11, map)                                                                              \
  X(TileX, "TileX", 1, arc::MAX_GRID_DIM, generate)                                                              \
  X(TileY, "TileY", 1, arc::MAX_GRID_DIM, generate)                                                              \
  /* Where a piece sits among the others, the one thing no other pick says. */   \
  X(SelectRow, "SelectRow", 0, arc::MAX_GRID_DIM - 1, select)                                                    \
  X(SelectColumn, "SelectColumn", 0, arc::MAX_GRID_DIM - 1, select)                                              \
  X(BidiColorMap, "BidiColorMap", 0, arc::COLOR_PAIR_COUNT - 1, map)                                             \
  /* The move IS the number of cells, so the whole grid is sayable in one. */    \
  X(MoveRow, "MoveRow", -arc::MAX_GRID_DIM, arc::MAX_GRID_DIM, translate)                                        \
  X(MoveColumn, "MoveColumn", -arc::MAX_GRID_DIM, arc::MAX_GRID_DIM, translate)

enum class node_type : uint8_t {
#define X(id, n, low, high, role) id,
    NODE_PRIMITIVES
#undef X
    type_count
};

// Every op's role, straight off the ROLE column. The one thing anyone reads to
// ask what an op does — nobody keeps a second list.
// A SWITCH, not a table: the kernel calls this (via is_get_op, in
// gpu_eval_program_warp), and a table would be host memory the device cannot
// reach. Relaxed-constexpr compiles that silently and it faults at runtime, so
// this must stay pure comparison with nothing to dereference.
inline constexpr op_role role_of(node_type t) {
    switch (t) {
#define X(id, n, low, high, role) case node_type::id: return op_role::role;
    NODE_PRIMITIVES
#undef X
    default: return op_role::map;
    }
}

// The canvas ops ("printer-ops"): the four that write ctx.output —
// Union/Underlay paint, Intersection/Difference erase. Everything else works on
// the pending object list. Kernel-verified 2026-07-12 (grep for ctx.output
// writes per exec function). Locks may only root at one of these: a lock
// certifies canvas cells, and only canvas paint survives into the continuations
// that build on the frozen prefix.
inline constexpr bool modifies_canvas(node_type t) {
    return t == node_type::Union || t == node_type::Underlay ||
           t == node_type::Intersection || t == node_type::Difference;
}

// --- Generation roles (the sentence: GET SELECT ACT COMBINE) --------------
// The four roles the generator's legality rules speak in, read off the ROLE
// column. GET = pure source (chain start). SELECT = marks objects, drops
// nothing. COMBINE = lands the list on the canvas. ACT = the four middle roles
// folded together — everything else that does work.
inline constexpr bool is_get_op(node_type t)     { return role_of(t) == op_role::get; }
inline constexpr bool is_select_op(node_type t)  { return role_of(t) == op_role::select; }
inline constexpr bool is_combine_op(node_type t) { return role_of(t) == op_role::combine; }
inline constexpr bool is_act_op(node_type t) {
    return !is_get_op(t) && !is_select_op(t) && !is_combine_op(t);
}

// How much of what an op does comes from a fact inference elected. ALL means the
// op is nothing without that fact, which is exactly when a dead-op guard is an
// honest verdict -- so the ALL ops are the ops that carry one, and that is the
// whole guard list. SOME means it reads the fact and still does real work with
// no fact to read: Split still segments, Colors still layers with the ground
// included, VoteDenoise falls back to a local background, and the seed pool just
// keeps the objects the ground would have taken out.
// The palette paints ALL yellow and SOME orange.
enum class inferred_amount { none, some, all };
inline constexpr inferred_amount how_much_is_inferred(node_type t) {
    switch (t) {
    case node_type::InputWithoutBackground:
    case node_type::SizeRankColorMap:
    case node_type::ShapeColorMap:
    case node_type::ShapeToInteriorColorMap:
    case node_type::Panes:                    return inferred_amount::all;
    case node_type::Colors:
    case node_type::Split:
    case node_type::VoteDenoise:
    case node_type::Seeds:
    case node_type::ProjectOnSeeds:           return inferred_amount::some;
    default:                                  return inferred_amount::none;
    }
}

// Carving clears selection so the next selector can choose individual pieces.
// Partition consumes selected walls; DeleteSelection keeps unselected survivors.
// These declarations must agree with the kernel implementations.
inline constexpr bool clears_the_marks(node_type t) {
    return t == node_type::Split || t == node_type::Colors ||
           t == node_type::Partition || t == node_type::DeleteSelection;
}

// Ops that do NOTHING when nothing is picked, so offering one before a pick is
// pure waste: it spells a program the search already has, one op longer. The
// generator refuses them until something is marked (legal_next).
// Each entry is kernel-checked, never guessed — Recolor and Shift pass unpicked
// objects through untouched, DeleteSelection deletes nothing, Partition cuts
// nothing without walls. Grows as more acts are taught to read the marks.
// NOT here: UnionList / IntersectionList, which do real work either way (with
// no marks they merge everything into one).
inline constexpr bool needs_a_pick(node_type t) {
    return t == node_type::Recolor || t == node_type::Shift ||
           t == node_type::DeleteSelection || t == node_type::Partition ||
           t == node_type::TurnRight || t == node_type::TurnLeft ||
           t == node_type::FlipHorizontal || t == node_type::FlipVertical ||
           t == node_type::MoveRow || t == node_type::MoveColumn ||
           t == node_type::SlamRow || t == node_type::SlamColumn ||
           t == node_type::MoveToEdge || t == node_type::Origin ||
           t == node_type::AlignToCanvas || t == node_type::Skew ||
           t == node_type::ScaleUp || t == node_type::ScaleDown ||
           t == node_type::Trim || t == node_type::Fit ||
           t == node_type::Outline ||
           t == node_type::InteriorColor || t == node_type::BidiColorMap ||
           t == node_type::VoteDenoise || t == node_type::ShapeToInteriorColorMap ||
           t == node_type::SortBySize ||
           t == node_type::TileX || t == node_type::TileY ||
           t == node_type::Fractal || t == node_type::AddColor ||
           t == node_type::Pattern || t == node_type::ExtendOrthogonal ||
           t == node_type::ExtendDiagonal ||
           // Taught to read the marks 2026-09-22, so they belong here now: with
           // nothing picked each one hands its whole list back untouched. Gating
           // them is what lets a chain keep every layer in hand and speak to one
           // at a time, instead of deleting the others to feed an op one thing.
           t == node_type::ExtendTo || t == node_type::ReadUnder ||
           t == node_type::MaskBy || t == node_type::MaskOut;
}

// Read a word's operation name; arguments remain the caller's concern.
// Unknown words (including the inference placeholder "?") have no operation.
inline constexpr node_type operation_named(std::string_view word) {
    word = word.substr(0, word.find('['));
#define X(id, name, low, high, role) if (word == name) return node_type::id;
    NODE_PRIMITIVES
#undef X
    return node_type::type_count;
}

// Visit space-separated program words without allocating temporary strings.
// Keep the original spelling so callers can reproduce arguments unchanged.
template <class See>
inline void each_program_word(std::string_view program, See see) {
    while (!program.empty()) {
        const auto first = program.find_first_not_of(' ');
        if (first == std::string_view::npos) break;
        program.remove_prefix(first);
        const auto end = program.find(' ');
        const auto word = program.substr(0, end);
        see(word, operation_named(word));
        if (end == std::string_view::npos) break;
        program.remove_prefix(end + 1);
    }
}
