// The words a program is written in, as 0_core.hpp holds the words a grid is.
// Numbered after 7_infer.hpp, whose elected facts some ops name. The vocabulary
// itself compiles against 0_core.hpp alone. Read by 10_sentences.hpp, which spells
// programs, and src/ops.cpp, which runs them, from the same definitions.
//
// WORDS THE READING CANNOT SAY YET (as of 2026-10-04). Inference says a word only
// after replaying it on every training pair, with the arithmetic the op runs,
// and seeing it at work in every pair. The ops below have no such
// replay in the general reading: no inferred program uses them, except where one
// special-case rule spells them for its own pattern. To unlock one:
//   1. Move the op's arithmetic into a shared helper in 9_op_math.hpp and
//      have the op in src/ops.cpp call it, so the replay and the op cannot
//      differ.
//   2. Replay it in 7_infer.hpp: a *_holds check on one of the replay templates
//      (paint_holds, slides_hold, drops_hold) asked in finish_the_line for a
//      same-size word, a frame kind in choose_piece_frame for a size change, or a
//      whole_line rule for a word that redraws every output by itself.
//   3. Spell it in 10_sentences.hpp (beats_of, drawn_after, frame_the_sentences).
//   4. Check with scripts\said.bat on the training set (no solved puzzle lost,
//      the wrong count not up) and the readouts' fully-explained-but-unsolved
//      list.
// Read what earlier paints left on the canvas, so they wait on composition:
//   Output, ReadUnder, Intersection, Difference (only the input-blocks and pane
//   rules spell it), AlignToCanvas (only slide-to-touch does). MaskBy and
//   MaskOut are said only after a paint -- rays, trails, copies, repeats --
//   laid only onto what is down (MaskBy[10]), or onto or off one color of it.
// IntersectionList: folds the picked pieces into one; only the pane rules say it.
// Seeds: every piece but the largest; only ProjectOnSeeds reads it, internally.
// Partition: cuts the parts the walls the selection marks fence off
//   (fence_parts, the parts SelectPartition picks by), each part's empty cells
//   handed back as a room for a recolor to paint; overlaps the pane rules.
// Panes: only the pane rules say it; it is not a frame kind yet.
// Fit: nearest-neighbor stretch to the input frame [0], output frame [1],
//   painted output bounds [2], or one pane's box [3] (fit_to, 9_op_math.hpp).
//   Pane-pick uses [0]; explicit background fill uses [1], [2], or [3] where
//   the answer is one pane; the filled frame uses [1] on
//   pieces of one color, filled to their boxes. General piece-stretch readings
//   still need a replay.
// NormalizeOrientation, RestoreOrientation: read every pair in one pose; only the
//   mark-lattice rule says them.
// SortBySize: said only as a helper where a word reads list order (size rank,
//   project on seeds); it changes nothing visible by itself.
// Skew: its range is -1..1, but the op keeps 47 slopes and -1 does nothing; fix
//   the range before it can be said.
#pragma once
#include "0_core.hpp"
#include <charconv>
#include <optional>
#include <string>
#include <string_view>

namespace arc {

constexpr int COLOR_PAIR_COUNT = COLOR_COUNT * (COLOR_COUNT - 1) / 2;

// Unordered color pairs, numbered {0,1}, {0,2}, ... {8,9}.
// Inference, program text, and the ops all use this same numbering.
struct color_pair {
    int first = TRANSPARENT, second = TRANSPARENT;

    constexpr int index() const {
        if (!is_color(first) || !is_color(second) || first == second) return -1;
        const int low = first < second ? first : second;
        const int high = first < second ? second : first;
        return low * (2 * COLOR_COUNT - low - 1) / 2 + high - low - 1;
    }

    static constexpr color_pair from_index(int index) {
        if (index < 0 || index >= COLOR_PAIR_COUNT) return {};
        for (int first = 0; first < COLOR_COUNT - 1; ++first) {
            const int remaining = COLOR_COUNT - first - 1;
            if (index < remaining) return { first, first + 1 + index };
            index -= remaining;
        }
        return {};
    }
};

// PAST ITS FIXED PAIRS, BidiColorMap trades the pairs a key drawn in the grid
// names (key_swaps, 9_op_math.hpp), read along the key's rows, or its columns;
// past those, one color with empty, EMPTY_PAIRS + the color (traded_with_empty).
constexpr int KEY_PAIRS_ALONG_ROWS = COLOR_PAIR_COUNT, KEY_PAIRS_ALONG_COLUMNS = COLOR_PAIR_COUNT + 1;
constexpr int EMPTY_PAIRS = KEY_PAIRS_ALONG_COLUMNS + 1;

// TILE'S CHOICES, A NAME EACH (P2, notes/2 - DSL/Design Principles.md). A program
// writes them slot by slot, Tile [steps: (0,3) (0,0), start: grid edge, end: grid edge,
// spread: lines], and that is their only form: tile_written writes it,
// tile_read reads it back, and each list below names its choices for both.
#define TILE_SLOTS X(steps) X(start) X(end) X(spread)
inline constexpr std::string_view tile_slot_names[] = {
#define X(name) #name,
    TILE_SLOTS
#undef X
};

// WHAT A STEP MAY MEASURE on each picture, so it grows with what it is read
// off: the piece's own width and own height, and the fewest empty rows or
// columns lying between two of its parts; and how many colors the picked
// pieces hold together.
#define TILE_MEASURES X(own_width, "own width") X(own_height, "own height") X(row_gap, "row gap") \
                      X(column_gap, "column gap") X(color_count, "color count")
enum class tile_measure : int8_t {
    none,
#define X(id, name) id,
    TILE_MEASURES
#undef X
};
inline constexpr std::string_view tile_measure_names[] = {
    "",
#define X(id, name) name,
    TILE_MEASURES
#undef X
};

// HOW THE COPIES GROW: in lines, each step taken from the piece alone, or into
// a grid, every copy taking every step again.
#define TILE_SPREADS X(lines) X(grid)
enum class tile_spread : int8_t {
#define X(name) name,
    TILE_SPREADS
#undef X
};
inline constexpr std::string_view tile_spread_names[] = {
#define X(name) #name,
    TILE_SPREADS
#undef X
};

// ONE PART OF A STEP: a count, plus or minus up to two measures -- "3", "own
// width", "-own width", "own height+row gap".
struct tile_number {
    static constexpr int MOST_MEASURES = 2;
    struct term {
        bool minus = false;
        tile_measure measure = tile_measure::none;
        bool operator==(const term&) const = default;
    };
    int8_t count = 0;
    term terms[MOST_MEASURES] = {};
    bool operator==(const tile_number&) const = default;
};

// A step part that is a measure, or two added.
constexpr tile_number measured(tile_measure first, tile_measure second = tile_measure::none) {
    return { .terms = { { false, first }, { false, second } } };
}

// WHERE THE COPIES BEGIN OR END: so many steps from the piece -- the piece
// itself 0, behind it negative -- or as far as the grid goes.
struct tile_bound {
    bool edge = false;
    int8_t steps = 0;
    bool operator==(const tile_bound&) const = default;
};

// WHAT A Tile SAYS: its two steps, each a move down and a move across, (0,0)
// for a step that goes nowhere; the first copy and the last; and how the
// copies grow.
struct tile_value {
    static constexpr int STEPS = 2;
    struct step {
        tile_number down, across;
        bool operator==(const step&) const = default;
    };
    step steps[STEPS] = {};
    tile_bound start, end;
    tile_spread spread = tile_spread::lines;
    bool operator==(const tile_value&) const = default;
};

inline std::string number_written(const tile_number& number) {
    std::string written;
    for (const tile_number::term& term : number.terms) {
        if (term.measure == tile_measure::none) continue;
        written += term.minus ? "-" : written.empty() ? "" : "+";
        written += tile_measure_names[(int)term.measure];
    }
    if (written.empty()) return std::to_string(number.count);
    if (number.count != 0) written += (number.count > 0 ? "+" : "") + std::to_string(number.count);
    return written;
}

inline std::string tile_written(const tile_value& tile) {
    auto bound = [](const tile_bound& at) { return at.edge ? std::string("grid edge") : std::to_string(at.steps); };
    std::string written = std::string(tile_slot_names[0]) + ":";
    for (const tile_value::step& step : tile.steps)
        written += " (" + number_written(step.down) + "," + number_written(step.across) + ")";
    return written + ", " + std::string(tile_slot_names[1]) + ": " + bound(tile.start) + ", " +
           std::string(tile_slot_names[2]) + ": " + bound(tile.end) + ", " + std::string(tile_slot_names[3]) + ": " +
           std::string(tile_spread_names[(int)tile.spread]);
}

// The whole of `text` as a whole number, or none.
inline std::optional<int> count_read(std::string_view text) {
    int count = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
    return count;
}

// A number written as number_written writes it, or none: counts and measures,
// each after a plus or a minus but the first, which may have a minus.
inline std::optional<tile_number> number_read(std::string_view text) {
    if (text.empty()) return std::nullopt;
    tile_number number;
    int measures = 0;
    for (bool first = true; !text.empty(); first = false) {
        const bool minus = text.front() == '-';
        if (minus || (!first && text.front() == '+')) text.remove_prefix(1);
        else if (!first) return std::nullopt;
        const auto next = text.find_first_of("+-");
        const std::string_view term = text.substr(0, next);
        if (const auto count = count_read(term)) number.count = (int8_t)(number.count + (minus ? -*count : *count));
        else {
            int measure = 1;
            while (measure < (int)std::size(tile_measure_names) && tile_measure_names[measure] != term) ++measure;
            if (measure == (int)std::size(tile_measure_names) || measures == tile_number::MOST_MEASURES)
                return std::nullopt;
            number.terms[measures++] = { minus, (tile_measure)measure };
        }
        text.remove_prefix(next == std::string_view::npos ? text.size() : next);
    }
    return number;
}

// A Tile's slots written as tile_written writes them, or none.
inline std::optional<tile_value> tile_read(std::string_view text) {
    tile_value tile;
    auto trimmed = [](std::string_view part) {
        while (!part.empty() && part.front() == ' ') part.remove_prefix(1);
        while (!part.empty() && part.back() == ' ') part.remove_suffix(1);
        return part;
    };
    // Each slot in turn, its name and then its value, up to the next comma
    // outside a step's parentheses.
    std::string_view slots[std::size(tile_slot_names)];
    for (size_t slot = 0; slot < std::size(tile_slot_names); ++slot) {
        text = trimmed(text);
        const std::string_view name = tile_slot_names[slot];
        if (!text.starts_with(name) || text.substr(name.size(), 1) != ":") return std::nullopt;
        text.remove_prefix(name.size() + 1);
        size_t end = 0;
        for (int depth = 0; end < text.size() && (depth > 0 || text[end] != ','); ++end)
            depth += text[end] == '(' ? 1 : text[end] == ')' ? -1 : 0;
        slots[slot] = trimmed(text.substr(0, end));
        text.remove_prefix(end == text.size() ? end : end + 1);
    }
    if (!trimmed(text).empty()) return std::nullopt;
    std::string_view steps = slots[0];
    for (tile_value::step& step : tile.steps) {
        const auto close = steps.find(')');
        const auto comma = steps.find(',');
        if (!steps.starts_with('(') || close == std::string_view::npos || comma > close) return std::nullopt;
        const auto down = number_read(trimmed(steps.substr(1, comma - 1)));
        const auto across = number_read(trimmed(steps.substr(comma + 1, close - comma - 1)));
        if (!down || !across) return std::nullopt;
        step = { *down, *across };
        steps = trimmed(steps.substr(close + 1));
    }
    if (!steps.empty()) return std::nullopt;
    auto bound = [](std::string_view written) -> std::optional<tile_bound> {
        if (written == "grid edge") return tile_bound{ true };
        const auto steps = count_read(written);
        if (!steps) return std::nullopt;
        return tile_bound{ false, (int8_t)*steps };
    };
    const auto start = bound(slots[1]), end = bound(slots[2]);
    int spread = 0;
    while (spread < (int)std::size(tile_spread_names) && tile_spread_names[spread] != slots[3]) ++spread;
    if (!start || !end || spread == (int)std::size(tile_spread_names)) return std::nullopt;
    tile.start = *start;
    tile.end = *end;
    tile.spread = (tile_spread)spread;
    return tile;
}

}  // namespace arc

// What an op does to the layer, in pipeline-authoring order (a single solution
// line reads left-to-right through these): bring a source in (GET) -> cut it
// down (REDUCE) -> pick objects (SELECT) -> move it (TRANSLATE) -> recolor or
// reason (MAP) -> grow it (GENERATE) -> merge into the canvas (COMBINE).
// A sentence's four roles fold out of this: the middle four are ACT.
// Order is frozen — the GUI indexes its tab rows by these ordinals.
enum class op_role { get, reduce, select, translate, map, generate, combine, count };

// How much an operation depends on inferred facts: none, some, or entirely.
enum class inferred_amount { none, some, all };

// --- Op vocabulary: the single home of the DSL op list ---
// The reading, the GUI, the serializer and the ops all derive from this one
// macro, so none can drift from the others.
// INFERRED says how much the op depends on learned facts.
// CLEARS means it removes selection marks. That describes what the op does
// (src/ops.cpp), so update both together.
// The ID also supplies its text spelling (#id), so names cannot drift.
// Every op is unary (one Objects input, one Objects output); nodes form a chain.
// ROLE is a column here, not a list elsewhere, so a new op cannot be added
// without saying what it does — the macro will not compile without it.
// AN OP'S ARGUMENT IS A RANGE, LOW AND HIGH, AND THAT IS THE ONLY PLACE IT IS
// SAID. How many values is HIGH - LOW + 1, which value is LOW + index, and
// whether the op takes an argument at all is LOW != HIGH -- three facts read
// off two numbers, so none of them can disagree with the others, and no index
// is interpreted anywhere. An op that takes nothing writes 0, 0.
#define NODE_PRIMITIVES /* Schema: X(ID, LOW, HIGH, ROLE, INFERRED, CLEARS) */                                   \
  X(SelectAll                   , 0                 , 0                        , select   , none, false)        \
  X(InvertSelection             , 0                 , 0                        , select   , none, false)        \
  X(TurnRight                   , 0                 , 0                        , translate, none, false)        \
  X(TurnLeft                    , 0                 , 0                        , translate, none, false)        \
  X(FlipHorizontal              , 0                 , 0                        , translate, none, false)        \
  X(FlipVertical                , 0                 , 0                        , translate, none, false)        \
  X(Input                       , 0                 , 0                        , get      , none, false)        \
  X(Output                      , 0                 , 0                        , get      , none, false)        \
  X(CurrentObjects              , 0                 , 0                        , get      , none, false)        \
  X(Origin                      , 0                 , 0                        , translate, none, false)        \
  X(Union                       , 0                 , 0                        , combine  , none, false)        \
  /* DeleteSelection consumes the marks, it does not make them: an act. */       \
  X(DeleteSelection             , 0                 , 0                        , reduce   , none, true )        \
  /* The canvas-readers judge by what lies beneath, not by the object: acts. */  \
  X(ReadUnder                   , 0                 , 0                        , map      , none, false)        \
  X(Outline                     , 0                 , 0                        , reduce   , none, false)        \
  X(SelectMostCommonColor       , 0                 , 0                        , select   , none, false)        \
  /* Colors extracts the palette as objects — a producing act, not a source. */  \
  X(Colors                      , 0                 , 0                        , map      , none, true )        \
  X(Trim                        , 0                 , 0                        , reduce   , none, false)        \
  X(OverlapCorner               , 0                 , 0                        , map      , none, false)        \
  X(Underlay                    , 0                 , 0                        , combine  , none, false)        \
  /* Withhold closes the beat without painting, so a sentence that only sets  */ \
  /* up the hand is sayable and does not read as damage to the canvas.        */ \
  X(Withhold                    , 0                 , 0                        , combine  , none, false)        \
  X(Fit                         , 0                 , 3                        , reduce   , none, false)        \
  X(Intersection                , 0                 , 0                        , combine  , none, false)        \
  X(Fractal                     , 0                 , 0                        , generate , none, false)        \
  X(Difference                  , 0                 , 0                        , combine  , none, false)        \
  X(UnionList                   , 0                 , 0                        , reduce   , none, false)        \
  X(AlignToCanvas               , 0                 , 0                        , translate, none, false)        \
  X(IntersectionList            , 0                 , 0                        , reduce   , none, false)        \
  X(SizeRankColorMap            , 0                 , 0                        , map      , all , false)        \
  X(Pattern                     , 0                 , 0                        , generate , none, false)        \
  X(ProjectOnSeeds              , 0                 , 0                        , map      , some, false)        \
  X(Partition                   , 0                 , 0                        , generate , none, true )        \
  X(RestoreOrientation          , 0                 , 0                        , translate, none, false)        \
  X(Seeds                       , 0                 , 0                        , get      , some, false)        \
  /* Partition cuts on the walls a selection marks; Panes cuts on the cut  */    \
  /* the split elected, which is the same act with a surer hand.           */        \
  X(Panes                       , 0                 , 0                        , generate , all , false)        \
  X(VoteDenoise                 , 0                 , 0                        , map      , some, false)        \
  X(ShapeToInteriorColorMap     , 0                 , 0                        , map      , all , false)        \
  X(SnapToAnchors               , 0                 , 0                        , map      , none, false)        \
  X(SortBySize                  , 0                 , 1                        , map      , none, false)        \
  X(NormalizeOrientation        , 0                 , 1                        , translate, none, false)        \
  /* Each picked piece takes the color its own key names: its shape [0] or  */ \
  /* its symmetry [1] (shape_key_of), or the color the table gives every    */ \
  /* key it does not name.                                                  */ \
  X(ShapeColorMap               , 0                 , 1                        , map      , all , false)        \
  /* A skew leans one way or the other, and 0 is the lean nobody takes. */       \
  X(Skew                        , -1                , 1                        , translate, none, false)        \
  /* Split[3] joins one color's cells by sides alone, the finest connected cut. */ \
  X(Split                       , 0                 , 3                        , generate , none, true )        \
  /* Cells slide a step and pile at the wall, so a step past the wall says */    \
  X(SlamColumn                  , -1                , 1                        , translate, none, false)        \
  /* nothing new: the answers run out at the object, not at the grid. */         \
  X(SlamRow                     , -1                , 1                        , translate, none, false)        \
  /* Tile says its choices in slots (tile_value), not in a number. */            \
  X(Tile                        , 0                 , 0                        , generate , none, false)        \
  X(SelectObjectsWithHoles      , 0                 , 4                        , select   , none, false)        \
  /* How a bridge reaches (bridge_bars): rows and columns, rows, or columns, */ \
  /* to the nearest piece, then the same three to every one, then to the one */ \
  /* facing it across the grid. Only the picked pieces are bridged, and only */ \
  /* the bars come back picked. Past those [9], an L from each picked piece  */ \
  /* to its one partner of each kind partner_of_kind names.                  */ \
  X(Bridge                      , 0                 , 31                       , generate , none, false)        \
  X(Shift                       , 0                 , 7                        , translate, none, false)        \
  X(MoveToEdge                  , 0                 , 7                        , translate, none, false)        \
  /* A trail one of the eight ways [0-7], or past them every picked piece    */ \
  /* grown in step with the others until they meet: along its rows, along    */ \
  /* its columns, to every side, along the canvas's edge [8-11]; along its   */ \
  /* diagonals one way or the other, both, its rows and columns, every way,  */ \
  /* to every side and corner [12-17] (grown_along).                        */ \
  X(ExtendTo                    , 0                 , 17                       , generate , none, false)        \
  /* Each picked piece wears its partner's color, as partner_by numbers them; */ \
  /* past those [13], the color a key drawn in the grid shows at its place;  */ \
  /* past that [14-19], its colors cell by cell from the one piece of its    */ \
  /* shape each likeness finds (colors_from_its_shape, 9_op_math.hpp).       */ \
  X(ColorFrom                   , 0                 , 19                       , map      , none, false)        \
  /* Of the picked pieces, only those that would fit into a gap of their      */ \
  /* partner stay picked (would_fit, 9_op_math.hpp). A pick, so it selects.  */ \
  X(WouldFit                    , 0                 , 12                       , select   , none, false)        \
  /* Selected patterns copied to unselected frames; 3x3 anchors, row-major. */ \
  X(CopyTo                      , 0                 , 8                        , generate , none, false)        \
  /* Rays as short as one cell -- a dot's four neighbours are rays too -- and */ \
  /* as long as a grid is wide, so a ray can run on to any edge.              */ \
  X(ExtendDiagonal              , 1                 , arc::MAX_GRID_DIM - 1    , generate , none, false)        \
  X(ExtendOrthogonal            , 1                 , arc::MAX_GRID_DIM - 1    , generate , none, false)        \
  X(ScaleUp                     , 2                 , 10                       , translate, none, false)        \
  X(ScaleDown                   , 2                 , 10                       , translate, none, false)        \
  /* Past the ten colors, BLANK: the rooms, Partition's empty pieces.        */ \
  X(SelectColor                 , 0                 , arc::BLANK               , select   , none, false)        \
  /* The one picked-from piece with no twin (unique_piece, 9_op_math.hpp):   */ \
  /* twins by drawing [0], shape as it lies [1], any way round [2], colors [3], */ \
  /* or past those the same count of what count_kind names [4-16].            */ \
  X(SelectUnique                , 0                 , 16                       , select   , none, false)        \
  X(SelectElectedBackground     , 0                 , 0                        , select   , none, false)        \
  /* The n-th piece by its cells [0-9]; past those, every piece with the most */ \
  /* (fewest, SelectSmallestObject) of what count_kind names (count_of).     */ \
  X(SelectLargestObject         , 0                 , 22                       , select   , none, false)        \
  X(SelectSmallestObject        , 0                 , 22                       , select   , none, false)        \
  X(AddColor                    , 0                 , 11                       , generate , none, false)        \
  X(MaskBy                      , 0                 , 10                       , map      , none, false)        \
  X(MaskOut                     , 0                 , 10                       , map      , none, false)        \
  /* Layout[0] reads its cell size off the pieces instead of being told it. */  \
  X(Layout                      , 0                 , 10                       , translate, none, false)        \
  X(InteriorColor               , -1                , arc::COLOR_COUNT - 1     , map      , none, false)        \
  X(Recolor                     , 0                 , 11                       , map      , none, false)        \
  /* Each picked piece moves level with its partner: rows first, then columns, */ \
  /* each by every way partner_by finds one.                                   */ \
  X(AlignTo                     , 0                 , 25                       , translate, none, false)        \
  /* Each picked piece drawn again mirrored around its partner's middle or the */ \
  /* grid's: left to right, top to bottom, all four ways, by mirror_by.        */ \
  X(MirrorAround                , 0                 , 41                       , generate , none, false)        \
  /* Where a piece sits among the others, the one thing no other pick says. */   \
  X(SelectRow                   , 0                 , arc::MAX_GRID_DIM - 1    , select   , none, false)        \
  X(SelectColumn                , 0                 , arc::MAX_GRID_DIM - 1    , select   , none, false)        \
  /* The picked pieces trade the colors of one fixed pair, color_pair's      */ \
  /* numbering, or past those, the pairs a key drawn in the grid names: the  */ \
  /* pieces not picked, read along their rows, then their columns; past      */ \
  /* those, one color with the empty cells of their frames.                  */ \
  X(BidiColorMap                , 0                 , arc::EMPTY_PAIRS + arc::COLOR_COUNT - 1, map, none, false) \
  /* The move IS the number of cells, so the whole grid is sayable in one. */    \
  X(MoveRow                     , -arc::MAX_GRID_DIM, arc::MAX_GRID_DIM        , translate, none, false)        \
  X(MoveColumn                  , -arc::MAX_GRID_DIM, arc::MAX_GRID_DIM        , translate, none, false)        \
  /* Toward a partner: how far (step, own size, until touching) x the partner, */ \
  /* each way partner_by finds one and then by each color (move_toward).      */ \
  X(MoveToward                  , 0                 , 68                       , translate, none, false)        \
  /* The pieces that paint at least so many cells (paints_at_least); fewer, */  \
  /* turned round by InvertSelection. One cell picks every piece, so 2 up.  */  \
  X(SelectAtLeastCells          , 2                 , 10                       , select   , none, false)        \
  /* The pieces on the row or column `from` lines past the middle of all    */ \
  /* they paint together (line_from_middle): 0 is the middle itself.        */ \
  X(SelectRowFromMiddle         , -14               , 14                       , select   , none, false)        \
  X(SelectColumnFromMiddle      , -14               , 14                       , select   , none, false)        \
  /* The picked pieces are walls, as Partition reads them, and the pick    */ \
  /* becomes the pieces lying in one part they fence off (fence_parts),   */ \
  /* the parts numbered in reading order, row by row, as rows wrap.       */ \
  X(SelectPartition             , 0                 , arc::MAX_GRID_DIM - 1    , select   , none, false)        \
  /* Of the picked pieces, only those like a piece of the key -- every    */ \
  /* piece not picked -- stay picked: of its shape as it lies or turned    */ \
  /* any way, at its size [0-1], any size [2-3] or stretched [4-5]          */ \
  /* (like_the_key, 9_op_math.hpp).                                        */ \
  X(SelectLikeKey               , 0                 , 5                        , select   , none, false)        \
  /* The pick moves onto every partner each picked piece has, the way       */ \
  /* partner_by finds it, lying on any side [0-12] or past those on each of */ \
  /* the eight directions, from up clockwise (each_partner_picked, 9_op_math). */ \
  X(SelectPartner               , 0                 , 116                      , select   , none, false)

enum class node_type : uint8_t {
#define X(id, low, high, role, inferred, clears) id,
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
    bool clears = false;
};

inline constexpr op_row row_of(node_type t) {
    switch (t) {
#define X(id, low, high, role, inferred, clears) \
    case node_type::id: return { low, high, op_role::role, inferred_amount::inferred, clears };
    NODE_PRIMITIVES
#undef X
    default: return {};
    }
}

// Every op's role, straight off the ROLE column, is the one thing anyone reads
// to ask what an op does.
inline constexpr op_role role_of(node_type t) { return row_of(t).role; }
inline constexpr bool clears_the_marks(node_type t) { return row_of(t).clears; }

// THE POSE EACH TURN OR FLIP SETS ITS PIECE IN, numbered as 1_readings.hpp
// numbers the eight: the op reads every cell through pose_source with it, and
// a replay through posed_place, its inverse, so the two cannot turn a piece
// differently. 0 for an op that turns nothing.
inline constexpr int pose_of(node_type t) {
    return t == node_type::TurnRight ? 3 : t == node_type::TurnLeft ? 5
         : t == node_type::FlipHorizontal ? 4 : t == node_type::FlipVertical ? 2 : 0;
}

// --- Sentence roles (GET SELECT ACT COMBINE) ------------------------------
// The four roles a sentence speaks in, read off the ROLE column. GET = pure
// source (chain start). SELECT = marks objects, drops nothing. COMBINE = lands
// the list on the canvas. ACT = the four middle roles folded together —
// everything else that does work.
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

// Whether an op says its choices in named slots (tile_value) rather than in a number.
inline constexpr bool said_in_slots(node_type t) { return t == node_type::Tile; }

// Read a word's operation name; arguments remain the caller's concern.
// Unknown words (including the inference placeholder "?") have no operation.
inline constexpr node_type operation_named(std::string_view word) {
    word = word.substr(0, word.find_first_of(" ["));
#define X(id, low, high, role, inferred, clears) if (word == #id) return node_type::id;
    NODE_PRIMITIVES
#undef X
    return node_type::type_count;
}

// Where the program word `text` opens with ends: at the first space outside
// its brackets, unless that space only parts its name from them, Recolor [3],
// Tile [steps: ...].
inline size_t program_word_end(std::string_view text) {
    size_t end = 0;
    for (int depth = 0; end < text.size(); ++end) {
        if (text[end] == ' ' && depth == 0 && text.substr(end + 1, 1) != "[") break;
        depth += text[end] == '[' ? 1 : text[end] == ']' ? -1 : 0;
    }
    return end;
}

// Visit space-separated program words without allocating temporary strings.
// Keep the original spelling so callers can reproduce arguments unchanged.
template <class See>
inline void each_program_word(std::string_view program, See see) {
    while (!program.empty()) {
        const auto first = program.find_first_not_of(' ');
        if (first == std::string_view::npos) break;
        program.remove_prefix(first);
        const size_t end = program_word_end(program);
        const auto word = program.substr(0, end);
        see(word, operation_named(word));
        if (end == program.size()) break;
        program.remove_prefix(end + 1);
    }
}
