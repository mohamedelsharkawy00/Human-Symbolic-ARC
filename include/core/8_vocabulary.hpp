// The words a program is written in, as 0_core.hpp holds the words a grid is.
// Numbered after 7_infer.hpp, whose elected facts some ops name. The vocabulary
// itself compiles against 0_core.hpp alone. Read by 9_sentences.hpp, which spells
// programs, and 12_dsl.cuh, which runs them, from the same definitions.
#pragma once
#include "0_core.hpp"
#include <string_view>

namespace arc {

constexpr int COLOR_PAIR_COUNT = COLOR_COUNT * (COLOR_COUNT - 1) / 2;

// Unordered color pairs, numbered {0,1}, {0,2}, ... {8,9}.
// Inference, program text, and CUDA all use this same numbering.
struct color_pair {
    int first = TRANSPARENT, second = TRANSPARENT;

    ARC_HOST_DEVICE constexpr int index() const {
        if (!is_color(first) || !is_color(second) || first == second) return -1;
        const int low = first < second ? first : second;
        const int high = first < second ? second : first;
        return low * (2 * COLOR_COUNT - low - 1) / 2 + high - low - 1;
    }

    ARC_HOST_DEVICE static constexpr color_pair from_index(int index) {
        if (index < 0 || index >= COLOR_PAIR_COUNT) return {};
        for (int first = 0; first < COLOR_COUNT - 1; ++first) {
            const int remaining = COLOR_COUNT - first - 1;
            if (index < remaining) return { first, first + 1 + index };
            index -= remaining;
        }
        return {};
    }
};

}  // namespace arc

// What an op does to the layer, in pipeline-authoring order (a single solution
// line reads left-to-right through these): bring a source in (GET) -> cut it
// down (REDUCE) -> pick objects (SELECT) -> move it (TRANSLATE) -> recolor or
// reason (MAP) -> grow it (GENERATE) -> merge into the canvas (COMBINE).
// The generator's four sentence roles fold out of this: the middle four are ACT.
// Order is frozen — the GUI indexes its tab rows by these ordinals.
enum class op_role { get, reduce, select, translate, map, generate, combine, count };

// How much an operation depends on inferred facts: none, some, or entirely.
enum class inferred_amount { none, some, all };

// --- Op vocabulary: the single home of the DSL op list ---
// CPU (search, GUI, serializer) and GPU (kernel dispatch) both derive from
// this one macro, so the two sides cannot drift apart.
// INFERRED says how much the op depends on learned facts.
// PICK means it does nothing without selected objects; CLEARS means it removes
// selection marks. These describe kernel behavior, so update both together.
// The ID also supplies its text spelling (#id), so names cannot drift.
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
#define NODE_PRIMITIVES /* Schema: X(ID, LOW, HIGH, ROLE, INFERRED, PICK, CLEARS) */                            \
  X(SelectAll                   , 0                 , 0                        , select   , none, false, false) \
  X(InvertSelection             , 0                 , 0                        , select   , none, false, false) \
  X(TurnRight                   , 0                 , 0                        , translate, none, true , false) \
  X(TurnLeft                    , 0                 , 0                        , translate, none, true , false) \
  X(FlipHorizontal              , 0                 , 0                        , translate, none, true , false) \
  X(FlipVertical                , 0                 , 0                        , translate, none, true , false) \
  X(Input                       , 0                 , 0                        , get      , none, false, false) \
  X(InputWithoutBackground      , 0                 , 0                        , get      , all , false, false) \
  X(Output                      , 0                 , 0                        , get      , none, false, false) \
  X(CurrentObjects              , 0                 , 0                        , get      , none, false, false) \
  X(Origin                      , 0                 , 0                        , translate, none, true , false) \
  X(Union                       , 0                 , 0                        , combine  , none, false, false) \
  /* DeleteSelection consumes the marks, it does not make them: an act. */       \
  X(DeleteSelection             , 0                 , 0                        , reduce   , none, true , true ) \
  /* The canvas-readers judge by what lies beneath, not by the object: acts. */  \
  X(ReadUnder                   , 0                 , 0                        , map      , none, true , false) \
  X(Outline                     , 0                 , 0                        , reduce   , none, true , false) \
  X(SelectMostCommonColor       , 0                 , 0                        , select   , none, false, false) \
  /* Colors extracts the palette as objects — a producing act, not a source. */  \
  X(Colors                      , 0                 , 0                        , map      , some, false, true ) \
  X(Trim                        , 0                 , 0                        , reduce   , none, true , false) \
  X(OverlapCorner               , 0                 , 0                        , map      , none, false, false) \
  X(Underlay                    , 0                 , 0                        , combine  , none, false, false) \
  /* Withhold closes the beat without painting, so a sentence that only sets  */ \
  /* up the hand is sayable and does not read as damage to the canvas.        */ \
  X(Withhold                    , 0                 , 0                        , combine  , none, false, false) \
  X(Fit                         , 0                 , 0                        , reduce   , none, true , false) \
  X(Intersection                , 0                 , 0                        , combine  , none, false, false) \
  X(Fractal                     , 0                 , 0                        , generate , none, true , false) \
  X(Difference                  , 0                 , 0                        , combine  , none, false, false) \
  X(UnionList                   , 0                 , 0                        , reduce   , none, false, false) \
  X(AlignToCanvas               , 0                 , 0                        , translate, none, true , false) \
  X(IntersectionList            , 0                 , 0                        , reduce   , none, false, false) \
  X(SizeRankColorMap            , 0                 , 0                        , map      , all , false, false) \
  X(Pattern                     , 0                 , 0                        , generate , none, true , false) \
  X(ProjectOnSeeds              , 0                 , 0                        , map      , some, false, false) \
  X(Bridge                      , 0                 , 0                        , generate , none, false, false) \
  X(Partition                   , 0                 , 0                        , generate , none, true , true ) \
  X(RestoreOrientation          , 0                 , 0                        , translate, none, false, false) \
  X(ShapeColorMap               , 0                 , 0                        , map      , all , false, false) \
  X(Seeds                       , 0                 , 0                        , get      , some, false, false) \
  /* Partition cuts on the walls a selection marks; Panes cuts on the cut  */    \
  /* the split elected, which is the same act with a surer hand.           */        \
  X(Panes                       , 0                 , 0                        , generate , all , false, false) \
  X(VoteDenoise                 , 0                 , 0                        , map      , some, true , false) \
  X(ShapeToInteriorColorMap     , 0                 , 0                        , map      , all , true , false) \
  X(SnapToAnchors               , 0                 , 0                        , map      , none, false, false) \
  X(SortBySize                  , 0                 , 1                        , map      , none, true , false) \
  X(DiagTile                    , 0                 , 1                        , generate , none, false, false) \
  X(NormalizeOrientation        , 0                 , 1                        , translate, none, false, false) \
  /* A skew leans one way or the other, and 0 is the lean nobody takes. */       \
  X(Skew                        , -1                , 1                        , translate, none, true , false) \
  X(Split                       , 0                 , 2                        , generate , some, false, true ) \
  /* Cells slide a step and pile at the wall, so a step past the wall says */    \
  X(SlamColumn                  , -1                , 1                        , translate, none, true , false) \
  /* nothing new: the answers run out at the object, not at the grid. */         \
  X(SlamRow                     , -1                , 1                        , translate, none, true , false) \
  X(Tile                        , 0                 , 3                        , generate , none, false, false) \
  X(SelectObjectsWithHoles      , 0                 , 4                        , select   , none, false, false) \
  X(Shift                       , 0                 , 7                        , translate, none, true , false) \
  X(MoveToEdge                  , 0                 , 7                        , translate, none, true , false) \
  X(ExtendTo                    , 0                 , 7                        , generate , none, true , false) \
  X(ExtendDiagonal              , 2                 , 10                       , generate , none, true , false) \
  X(ExtendOrthogonal            , 2                 , 10                       , generate , none, true , false) \
  X(ScaleUp                     , 2                 , 10                       , translate, none, true , false) \
  X(ScaleDown                   , 2                 , 10                       , translate, none, true , false) \
  X(Layout                      , 2                 , 10                       , translate, none, false, false) \
  X(SelectColor                 , 0                 , arc::COLOR_COUNT - 1     , select   , none, false, false) \
  X(SelectLargestObject         , 0                 , 9                        , select   , none, false, false) \
  X(SelectSmallestObject        , 0                 , 9                        , select   , none, false, false) \
  X(AddColor                    , 0                 , arc::COLOR_COUNT - 1     , generate , none, true , false) \
  X(MaskBy                      , 0                 , 10                       , map      , none, true , false) \
  X(MaskOut                     , 0                 , 10                       , map      , none, true , false) \
  X(InteriorColor               , -1                , arc::COLOR_COUNT - 1     , map      , none, true , false) \
  X(Recolor                     , 0                 , 11                       , map      , none, true , false) \
  X(TileX                       , 1                 , arc::MAX_GRID_DIM        , generate , none, true , false) \
  X(TileY                       , 1                 , arc::MAX_GRID_DIM        , generate , none, true , false) \
  /* Where a piece sits among the others, the one thing no other pick says. */   \
  X(SelectRow                   , 0                 , arc::MAX_GRID_DIM - 1    , select   , none, false, false) \
  X(SelectColumn                , 0                 , arc::MAX_GRID_DIM - 1    , select   , none, false, false) \
  X(BidiColorMap                , 0                 , arc::COLOR_PAIR_COUNT - 1, map      , none, true , false) \
  /* The move IS the number of cells, so the whole grid is sayable in one. */    \
  X(MoveRow                     , -arc::MAX_GRID_DIM, arc::MAX_GRID_DIM        , translate, none, true , false) \
  X(MoveColumn                  , -arc::MAX_GRID_DIM, arc::MAX_GRID_DIM        , translate, none, true , false)

enum class node_type : uint8_t {
#define X(id, low, high, role, inferred, pick, clears) id,
    NODE_PRIMITIVES
#undef X
    type_count
};

// ONE OP'S ROW of the table above, so a question asked of an op reads the
// table and no second list can disagree with it. An id outside the table gets
// the default row, whose empty range (low above high) takes no value.
struct op_row {
    int low = 1, high = 0;
    op_role role = op_role::map;
    inferred_amount inferred = inferred_amount::none;
    bool pick = false, clears = false;
};

// SWITCHES, not tables: the kernel calls these, and a table would be host
// memory the device cannot reach. Relaxed-constexpr compiles that silently and
// it faults at runtime, so they must stay pure comparison with nothing to
// dereference.
inline constexpr op_row row_of(node_type t) {
    switch (t) {
#define X(id, low, high, role, inferred, pick, clears) \
    case node_type::id: return { low, high, op_role::role, inferred_amount::inferred, pick, clears };
    NODE_PRIMITIVES
#undef X
    default: return {};
    }
}

// THE TWO COLUMNS THE KERNEL READS on every op it runs (gpu_eval_program_warp)
// keep a switch of their own. Read through row_of, the whole row cost that
// function registers: 72 more bytes of spill stores and 176 of loads, measured
// 2026-09-24. Every op's role, straight off the ROLE column, is the one thing
// anyone reads to ask what an op does.
inline constexpr op_role role_of(node_type t) {
    switch (t) {
#define X(id, low, high, role, inferred, pick, clears) case node_type::id: return op_role::role;
    NODE_PRIMITIVES
#undef X
    default: return op_row{}.role;
    }
}
inline constexpr bool clears_the_marks(node_type t) {
    switch (t) {
#define X(id, low, high, role, inferred, pick, clears) case node_type::id: return clears;
    NODE_PRIMITIVES
#undef X
    default: return op_row{}.clears;
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

// Whether an op's argument can be this value, off the op's own row above. A
// sentence that reads a number off the puzzle asks here before writing it, so
// it never spells an argument the op has no variant for.
inline constexpr bool takes_value(node_type t, int value) {
    return row_of(t).low <= value && value <= row_of(t).high;
}

// Properties come from the op's row, like its role and argument range.
inline constexpr inferred_amount how_much_is_inferred(node_type t) { return row_of(t).inferred; }
inline constexpr bool needs_a_pick(node_type t) { return row_of(t).pick; }

// Read a word's operation name; arguments remain the caller's concern.
// Unknown words (including the inference placeholder "?") have no operation.
inline constexpr node_type operation_named(std::string_view word) {
    word = word.substr(0, word.find('['));
#define X(id, low, high, role, inferred, pick, clears) if (word == #id) return node_type::id;
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
