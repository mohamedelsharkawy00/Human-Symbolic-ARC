#pragma once
#include "13_environment.hpp"
#include "9_sentences.hpp"  // sentences_of: the pane says a program the way the log does
#include "23_grid_buffer.hpp"
#include "24_global_app_state.hpp"
#include "26_vulkan_context.hpp"
#include "27_sound_manager.hpp"

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace arc {
using namespace std;
// 3x5 Bitmap Font (Each char is 5 octal digits)
inline constexpr string_view font_data =
    "A75755B65656C74447D65556E74747F74644G74557H55755I72227J11157K55655L44447"
    "M57555N75555O75557P75744Q75531R75655S74717T72222U55557V55552W55575X55255"
    "Y55222Z71247075557126227271747371717455711574717674757771111875757975717"
    ":02020.00002+02720(24442)21112<12421>42124,00224_00007/11244 00000[74447]71117-00700";

constexpr int UI_COLOR_LINE = COLOR_COUNT;
constexpr int UI_COLOR_PASTEL_START = COLOR_COUNT * 2;

inline display_grid create_character_grid(uint32_t ch) {
  static map<uint32_t, int> glyphs;
  if (glyphs.empty()) {
    for (size_t i = 0; i < font_data.size(); i += 6) {
      char c = font_data[i];
      int v = 0;
      from_chars(font_data.data() + i + 1, font_data.data() + i + 6, v, 8);
      glyphs[toupper((unsigned char)c)] = v;
      glyphs[tolower((unsigned char)c)] = v;
    }
    // Hardcoded special symbols
    glyphs[0x25B2] = 027777; // (Solid Triangle Up)
    glyphs[0x25BC] = 077772; // (Solid Triangle Down)
    glyphs['|']    = 022222; // (Vertical Bar: middle column lit every row)
    // Mute-toggle icons (3x5 each). Private-use codepoints so they never
    // collide with text. The muted state renders the cone followed by the X in
    // the next cell, giving a two-cell speaker-with-X (no per-glyph width
    // support needed in the font pipeline).
    glyphs[0xE000] = 017771; // SPEAKER right: rows 001 111 111 111 001 (taller left base, mouth flares right)
    glyphs[0xE001] = 055255; // X overlay    : rows 101 101 010 101 101
    glyphs[0xE002] = 067773; // COPY  : two overlapping pages (back top-left corner, front bottom-right corner)
    glyphs[0xE003] = 027557; // PASTE : clipboard (clip tab on top, board below)
  }
  display_grid g;
  g.height = 5;
  g.width = 3;
  g.at.stride = 3;
  if (glyphs.count(ch)) {
    int v = glyphs[ch];
    for (int row_idx = 0; row_idx < 5; ++row_idx) {
      int row_bits = (v >> (3 * (4 - row_idx))) & 7;
      for (int col_idx = 0; col_idx < 3; ++col_idx) {
        g.data[row_idx][col_idx] = (row_bits & (1 << (2 - col_idx))) ? UI_COLOR_LINE : 0;
      }
    }
  }
  return g;
}

// Global cache for colored fonts: [ColorIdx][Codepoint] -> Grid
inline map<int, map<uint32_t, display_grid>> colored_fonts;
// Global cache for solid color sheets: [ColorIdx] -> Grid
inline map<int, display_grid> solid_sheets;

inline int utf8_len(const string &s) {
  int len = 0;
  for (size_t i = 0; i < s.size(); ) {
    unsigned char c = (unsigned char)s[i];
    if (c >= 0xF0) i += 4;
    else if (c >= 0xE0) i += 3;
    else if (c >= 0xC0) i += 2;
    else i += 1;
    len++;
  }
  return len;
}

// Visual pixel width of text (each glyph is 3px wide + 1px gap, minus trailing gap)
inline int text_w(const string &s, int scale) {
  return utf8_len(s) * 4 * scale - scale;
}

inline void draw_text(vulkan_context *context, const map<uint32_t, display_grid> &font,
                      const string &text, int x, int y, int scale) {
  int cur_x = x;
  for (size_t i = 0; i < text.size(); ) {
    unsigned char c = (unsigned char)text[i];
    uint32_t cp = c;
    int len = 1;
    if (c >= 0xE0 && i + 2 < text.size()) {
        cp = ((c & 0x0F) << 12) | (((unsigned char)text[i+1] & 0x3F) << 6) | ((unsigned char)text[i+2] & 0x3F);
        len = 3;
    } else if (c >= 0xC0 && i + 1 < text.size()) {
        cp = ((c & 0x1F) << 6) | ((unsigned char)text[i+1] & 0x3F);
        len = 2;
    }

    if (font.count(cp)) {
      const display_grid &g = font.at(cp);
      context->draw(g.at.offset, g.width, g.height, cur_x, y, scale, g.at.stride);
      cur_x += (g.width + 1) * scale;
    } else if (cp == ' ') {
      cur_x += 4 * scale;
    } else {
      uint32_t up = toupper((unsigned char)cp);
      if (font.count(up)) {
        const display_grid &g = font.at(up);
        context->draw(g.at.offset, g.width, g.height, cur_x, y, scale, g.at.stride);
        cur_x += (g.width + 1) * scale;
      } else {
        cur_x += 4 * scale;
      }
    }
    i += len;
  }
}

// Fill a rectangle with one solid color. The solid sheets are only 256 wide, so
// a wide fill has to go out in tiles with stride 1: in one call it reads off the
// end of its own sheet and paints the neighbouring ones as diagonal streaks.
// Every fill wider than a glyph goes through here, so that cannot be forgotten.
inline void fill_rect(vulkan_context *context, int color, int x, int y, int w, int h) {
  if (w <= 0 || h <= 0 || !solid_sheets.count(color)) return;
  int off = solid_sheets[color].at.offset;
  const int tile = 200;   // comfortably inside the sheet
  for (int fx = 0; fx < w; fx += tile)
    context->draw(off, min(tile, w - fx), h, x + fx, y, 1, 1);
}

// Draws a centered text label inside a colored background box.
inline void draw_boxed_text(vulkan_context *context,
                            const map<uint32_t, display_grid> &font, const string &text,
                            int x, int y, int scale, int box_w, int box_h,
                            int color_idx) {
  // Retrieve Pre-allocated Solid Sheets
  int pastel_idx = UI_COLOR_PASTEL_START + color_idx;
  const display_grid *bg_grid = solid_sheets.count(pastel_idx)
                            ? &solid_sheets[pastel_idx]
                            : &solid_sheets[color_idx];

  context->draw(bg_grid->at.offset, box_w, box_h, x, y, 1, bg_grid->at.stride);

  int text_x = x + (box_w - text_w(text, scale)) / 2;
  int text_y = y + (box_h - 5 * scale) / 2;

  const map<uint32_t, display_grid> &box_font =
      colored_fonts.count(pastel_idx) ? colored_fonts[pastel_idx] : font;
  draw_text(context, box_font, text, text_x, text_y, scale);
}

inline string get_node_label(node_type type, int16_t value) {
  if (to_underlying(type) < 0 ||
      to_underlying(type) >= to_underlying(node_type::type_count))
    return "?";
  string label = type_name[to_underlying(type)];
  append_annotation(type, value, label);
  return label;
}

// --- draw_frame decomposition ---

struct frame_layout {
  int base_margin, scale, v_gap;
  int row1_y;
  // Set once the header has drawn and knows how many lines it needed.
  int grid_labels_y, grid_data_start_y;
  // The footer rows. The grids end where the BEST row starts.
  int best_y, elapsed_y;
  int h_gap_internal;
  int screen_w, screen_h;
};

inline frame_layout compute_frame_layout(int screen_w, int screen_h) {
  frame_layout l{};
  l.screen_w = screen_w;
  l.screen_h = screen_h;
  l.base_margin = screen_w / 200;
  // SCALE FROM THE SCREEN, NOT A CONSTANT. Every distance in the frame is a
  // count of characters times this, so one number fixes whether the whole
  // layout fits — and at a hardcoded 3 the pane ran off the bottom of a 1080p
  // screen while sitting comfortably on a 1440p one. The frame was laid out at
  // 2560x1440, so hold that ratio and take whichever axis is tighter. Whole
  // numbers only: the font is a bitmap and a fractional one is a smear.
  l.scale = max(1, min(screen_w * 3 / 2560, screen_h * 3 / 1440));
  l.v_gap = 2 * l.scale;
  l.h_gap_internal = 12 * l.scale;

  int row_h = 9 * l.scale + l.v_gap;

  l.row1_y = l.base_margin;

  // Footer: BEST, ELAPSED top-down
  l.elapsed_y = screen_h - l.base_margin - 9 * l.scale;
  l.best_y = l.elapsed_y - row_h;

  return l;
}

// Walk a unary program tree into execution order (terminal leaf -> root), the
// same order the footer pipeline draws its tokens.
inline void collect_pipeline_ops(node *root, vector<node *> &ops) {
  ops.clear();
  for (node *cur = root; cur; cur = cur->arg)
    ops.push_back(cur);
  reverse(ops.begin(), ops.end());
}

// A LONG PIPELINE FOLDS to the four lines the footer always keeps for it: its
// first ops, then a box saying how many more, which opens every line; open, a
// box after its last op folds it again (show_whole_programs, the facts pane's
// fold too). Every line at once ran BEST off the bottom of the screen
// (31aa019c's 79 ops). The boxes a pipeline of `ops` draws:
// `shown` ops, and the fold's box when it has one.
constexpr size_t PIPELINE_FOLDED_LINES = 4;
struct pipeline_fold { size_t shown; bool toggle; };
inline pipeline_fold fold_of(size_t ops, bool whole) {
  constexpr size_t room = PIPELINE_FOLDED_LINES * 4;
  if (ops <= room) return { ops, false };
  return { whole ? ops : room - 1, true };
}

// A program is sentences, so it reads as sentences: one to a line, four beats to
// a sentence, and each beat in a column as wide as the widest word that lands in
// it. The columns are what make GET under GET and ACT under ACT.
// One line per sentence. MUST agree with draw_pipeline or reflow and draw
// disagree about where the block ends: both count the boxes fold_of says.
inline int count_pipeline_lines(node *root, bool whole) {
  vector<node *> ops;
  collect_pipeline_ops(root, ops);
  if (ops.empty()) return 1;
  const pipeline_fold fold = fold_of(ops.size(), whole);
  return (int)((fold.shown + fold.toggle + 3) / 4);
}

// Draws the header: every control and stat, in reading order. Returns the
// bottom of the last line it needed, so what sits under it knows where to start.
inline int draw_header_rows(
    vulkan_context *context,
    const map<uint32_t, display_grid> &font,
    global_app_state &app_state,
    const frame_layout &lay,
    const puzzle &task,
    int puzzle_idx, int puzzle_count, float fps) {

  int scale = lay.scale;

  // Format display strings.
  // Width the index/count to the digit-count of puzzle_count so the value never
  // outgrows its reserved slot (e.g. 1000/1000 on the training set, which used
  // to overflow the fixed 3-digit reservation and shove the >/>| buttons onto
  // the number).
  int puz_digits = (int)to_string(puzzle_count > 0 ? puzzle_count : 1).size();
  string puz_str = to_text("%0*d/%0*d", puz_digits, puzzle_idx + 1, puz_digits, puzzle_count);
  string fps_str = to_text("%06.1f", fps);
  string status_str = string(app_state.is_paused ? "PAUSED" : "RUNNING");
  string log_state_str = (app_state.log_mode == logging_mode::stream) ? "ON" : "OFF";

  int pad = 2 * scale; // padding on each side of boxed text
  int box_h = 5 * scale + pad * 2;

  // The puzzle counter's slot, widened to the digits the set actually needs so
  // the value never outgrows it and shoves the nav buttons onto the number.
  int puz_val_w = text_w(string(puz_digits, '8') + "/" + string(puz_digits, '8'), scale);

  // The whole header is one flow: every piece asks for a spot as wide as it
  // needs, and the line wraps only when the next piece would not fit. Nothing is
  // ever stretched -- a line is filled by carrying more, not by spreading gaps,
  // so a wide screen buys extra pieces per line rather than extra air.
  int row_h = 9 * scale + lay.v_gap;
  int right_edge = lay.screen_w - lay.base_margin;
  int fx = lay.base_margin, fy = lay.row1_y;

  // Hands back the left edge for a piece `w` wide, wrapping first if it would
  // not fit on what is left of this line. Its top edge is `fy`.
  auto spot = [&](int w) {
    if (fx > lay.base_margin && fx + w > right_edge) { fx = lay.base_margin; fy += row_h; }
    int x = fx;
    fx += w + pad;
    return x;
  };

  // Width of a box holding this text.
  auto box_w = [&](const string &s) { return text_w(s, scale) + pad * 2; };

  // One boxed thing at a spot already chosen; `none` means it is not clickable.
  auto box_at = [&](const string &s, int x, int y, int w, int color, ui_action act) {
    draw_boxed_text(context, font, s, x, y, scale, w, box_h, color);
    if (act != ui_action::none)
      app_state.click_zones.push_back({x, y, w, box_h, act});
  };

  // A button sizes itself to its text and lights up while its action is active.
  auto button_at = [&](const string &s, int x, int y, ui_action act, int color) {
    int w = box_w(s);
    box_at(s, x, y, w, (app_state.active_ui_action == act) ? 4 : color, act);
    return w;
  };

  // A dim name box with its value written beside it. Every value that changes is
  // padded to its widest, so nothing shifts as the numbers move.
  auto pair_piece = [&](const string &name, const string &value) {
    int nw = box_w(name);
    int x = spot(nw + (value.empty() ? 0 : pad + text_w(value, scale))), y = fy;
    box_at(name, x, y, nw, 5, ui_action::none);
    if (!value.empty())
      draw_text(context, font, value, x + nw + pad, y + pad, scale);
  };

  // A button standing on its own in the flow.
  auto button_piece = [&](const string &name, const string &value, ui_action act, int color) {
    string full = value.empty() ? name : name + " " + value;
    button_at(full, spot(box_w(full)), fy, act, color);
  };

  // NAME [<] value [>]. The value sits between its two buttons, so the three ask
  // for one spot together and a wrap can never land in the middle of them.
  auto stepper_piece = [&](const string &name, const string &value, int value_w,
                           ui_action down, ui_action up) {
    int nw = box_w(name), aw = box_w("<");
    int x = spot(nw + pad + aw + pad + value_w + pad + aw), y = fy;
    box_at(name, x, y, nw, 5, ui_action::none);
    x += nw + pad;
    x += button_at("<", x, y, down, 1) + pad;
    draw_text(context, font, value, x, y + pad, scale);
    x += value_w + pad;
    button_at(">", x, y, up, 1);
  };
  button_piece("EXIT [ESC]", "", ui_action::exit_app, 1);

  // Mute toggle: speaker icon tucked right after EXIT. Glyph swaps on/muted
  // (private-use codepoints, UTF-8 encoded); muted also flips to the active
  // color so the state reads at a glance. Also bound to the [M] key.
  {
    bool muted = sound_manager::get().is_muted();
    const char *speaker = "\xEE\x80\x80"; // U+E000
    const char *cross   = "\xEE\x80\x81"; // U+E001
    string icon = muted ? string(speaker) + cross : speaker;
    // A fixed 2-glyph box whatever the state, and box_at centers the icon in it,
    // so the lone speaker lands in the middle and the width never moves.
    int w = box_w("88");
    int color = (app_state.active_ui_action == ui_action::toggle_mute || muted) ? 4 : 1;
    box_at(icon, spot(w), fy, w, color, ui_action::toggle_mute);
  }

  button_piece("MINIMIZE", "", ui_action::minimize_app, 1);

  // The inference overlay's on/off sits with the buttons, not down in the pane:
  // at OFF the pane isn't drawn, so a button inside it could never bring it back.
  {
    static const char *view_names[] = { "OFF", "FACTS", "LINES" };
    int view = app_state.inference_view.load() % global_app_state::INFERENCE_STEPS;
    button_piece("INFERENCE", to_text("%-5s [I]", view_names[view]),
                 ui_action::cycle_inference, view ? 4 : 1);
  }

  button_piece("RESET [R]", "", ui_action::reset_solver, 1);
  button_piece("LOGGING", to_text("%-3s [L]", log_state_str.c_str()), ui_action::toggle_log, 1);
  button_piece(to_text("%-7s [SPACE]", status_str.c_str()), "", ui_action::toggle_pause, 1);
  button_piece("IMPORT [N]", "", ui_action::import_sol, 1);
  button_piece("EXPORT [E]", "", ui_action::export_sol, 1);
  button_piece(to_text("BENCHMARK %-2s [B]", app_state.is_benchmarking ? "ON" : ""), "",
               ui_action::toggle_benchmark, app_state.is_benchmarking ? 4 : 1);
  stepper_piece("LIMIT", to_text("%03d", app_state.benchmark_puzzle_limit),
                text_w("888", scale), ui_action::dec_limit, ui_action::inc_limit);

  // PUZZLE [|<] [<] 001/600 [>] [>|], one piece for the same reason as a stepper.
  {
    int nw = box_w("PUZZLE"), aw = box_w("<"), ew = box_w("|<");
    int x = spot(nw + pad + ew + pad + aw + pad + puz_val_w + pad + aw + pad + ew), y = fy;
    box_at("PUZZLE", x, y, nw, 5, ui_action::none);
    x += nw + pad;
    x += button_at("|<", x, y, ui_action::first_puzzle, 1) + pad;
    x += button_at("<", x, y, ui_action::prev_puzzle, 1) + pad;
    draw_text(context, font, puz_str, x, y + pad, scale);
    x += puz_val_w + pad;
    x += button_at(">", x, y, ui_action::next_puzzle, 1) + pad;
    button_at(">|", x, y, ui_action::last_puzzle, 1);
  }

  // Dataset checkmarks: TRN / EVL / TST / CUS. Each toggles whether that set's tasks
  // are concatenated into the loaded list; clicking reloads the merged list live
  // (handled by the worker). On = lit box with a '+' prefix, off = dim with '-'.
  // (The font has '+'/'-' but no tick glyph, so the +/- reads the state.)
  {
    int ds = app_state.dataset_flags.load();
    int w = box_w("+TRN");
    int x = spot(w * 4 + pad * 3), y = fy;
    auto draw_ds = [&](const char *name, int flag, ui_action act) {
      bool on = (ds & flag) != 0;
      box_at(string(on ? "+" : "-") + name, x, y, w, on ? 4 : 1, act);
      x += w + pad;
    };
    draw_ds("TRN", 1 /*DS_TRAIN*/, ui_action::toggle_ds_train);
    draw_ds("EVL", 2 /*DS_EVAL*/,  ui_action::toggle_ds_eval);
    draw_ds("TST", 4 /*DS_TEST*/,  ui_action::toggle_ds_test);
    draw_ds("CUS", 8 /*DS_CUSTOM*/, ui_action::toggle_ds_custom);
  }

  // Clickable ID field: click to edit (type/paste an ID + Enter to jump), and it
  // doubles as a copy button via Ctrl+C. The value box renders the edit buffer
  // with a cursor while editing, otherwise the current ID.
  {
    string copy_lbl = string("\xEE\x80\x82") + " [CTRL C]";  // copy glyph U+E002
    string paste_lbl = string("\xEE\x80\x83") + " [CTRL V]"; // paste glyph U+E003
    int nw = box_w("ID");
    // A stable width so the click target doesn't jump around as the typed text
    // grows ("888..." is wider than any real 8-char ARC id).
    int vw = box_w("88888888");
    int x = spot(nw + pad + vw + pad + box_w(copy_lbl) + pad + box_w(paste_lbl)), y = fy;
    box_at("ID", x, y, nw, 5, ui_action::none);
    x += nw + pad;

    bool editing;
    bool error;
    string shown;
    {
      std::lock_guard<std::mutex> lk(app_state.id_edit_mtx);
      editing = app_state.id_edit.active;
      error = app_state.id_edit.error;
      shown = editing ? app_state.id_edit.buffer + "_" : task.id;
    }
    // Box the value while editing (or on error) so it reads as an input.
    if (editing || error)
      box_at(shown, x, y, vw, error ? 4 : 5, ui_action::edit_id);
    else {
      draw_text(context, font, shown, x + pad, y + pad, scale);
      app_state.click_zones.push_back({x, y, vw, box_h, ui_action::edit_id});
    }
    x += vw + pad;

    // Both also answer to their Ctrl key from anywhere, so both wear it.
    x += button_at(copy_lbl, x, y, ui_action::copy_id, 1) + pad;
    button_at(paste_lbl, x, y, ui_action::paste_id, 1);
  }

  pair_piece("FPS", to_text("%-7s", fps_str.c_str()));

  // ARC palette reference -- a flush strip of square swatches, each with its
  // index drawn with a black outline.
  {
    int n_colors = COLOR_COUNT;
    // The white outline lives inside the line's box: swatches are inset by t so
    // the strip's outer edge lands exactly on the line height.
    int t = max(1, scale / 2);
    int sw_side = box_h - t * 2;   // square swatch, inside the outline
    int strip_w = n_colors * sw_side + t * 2;
    int x = spot(strip_w), y = fy;
    // White outline around the whole strip; the swatches drawn on top leave only
    // the outer rim visible.
    fill_rect(context, UI_COLOR_LINE, x, y, strip_w, box_h);
    for (int i = 0; i < n_colors; ++i) {
      int cx = x + t + i * sw_side;
      int cy = y + t;
      if (solid_sheets.count(i)) {
        const display_grid &sw = solid_sheets[i];
        context->draw(sw.at.offset, sw_side, sw_side, cx, cy, 1, sw.at.stride);
      }
      // Index digit: white with a black outline, centered on the swatch.
      string num = to_text("%d", i);
      int nx = cx + (sw_side - text_w(num, scale)) / 2;
      int ny = cy + (sw_side - 5 * scale) / 2;
      // Stamp the digit in black at the 8 neighbor offsets, then white on top.
      if (colored_fonts.count(0)) {
        const auto &black_font = colored_fonts[0];
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            if (dx || dy)
              draw_text(context, black_font, num, nx + dx * scale, ny + dy * scale, scale);
      }
      const auto &white_font = colored_fonts.count(UI_COLOR_LINE) ? colored_fonts[UI_COLOR_LINE] : font;
      draw_text(context, white_font, num, nx, ny, scale);
    }
  }

  return fy + row_h;
}

// Draws the bottom of the screen: the BEST program pipeline and the elapsed
// timer.
inline void draw_footer_rows(
    vulkan_context *context,
    const map<uint32_t, display_grid> &font,
    global_app_state &app_state,
    const frame_layout &lay,
    const puzzle &task,
    float elapsed) {

  int scale = lay.scale;
  int pad = 2 * scale;

  // Walk a unary tree from the root op outward (Input first), pushing each
  // op as its own colored token. Returns end x.
  // `top_y` is the first line; long pipelines wrap downward at the right edge
  // (continuation lines indent under the first token). Footer Y-positions are
  // reflowed in draw_frame so the wrapped block stacks without overlapping.
  auto draw_pipeline = [&](node *root, int start_x, int top_y) {
    int box_h = 5 * scale + pad * 2;
    int gap = pad;
    int max_x = lay.screen_w - lay.base_margin;
    int line_h = 9 * scale + lay.v_gap; // matches frame_layout row_h
    vector<node *> ops;
    collect_pipeline_ops(root, ops);
    (void)max_x;  // sentences decide the wrap now, not the right edge

    uint64_t hh = app_state.hovered_hash.load();
    int x = start_x, y = top_y, end_x = start_x;
    const bool whole = app_state.show_whole_programs.load();
    const pipeline_fold fold = fold_of(ops.size(), whole);
    for (size_t i = 0; i < fold.shown + fold.toggle; ++i) {
      // A new sentence starts a new line; inside one, words just follow on.
      if (i > 0 && i % 4 == 0) { x = start_x; y += line_h; }
      if (i == fold.shown) {
        // The fold's own box, pressed like any button: how many more, or FOLD.
        const string label = whole ? string("FOLD") : to_text("... %d MORE", (int)(ops.size() - fold.shown));
        int bw = text_w(label, scale) + pad * 2;
        draw_boxed_text(context, font, label, x, y, scale, bw, box_h,
                        app_state.active_ui_action == ui_action::toggle_programs ? 4 : 5);
        app_state.click_zones.push_back({x, y, bw, box_h, ui_action::toggle_programs});
        x += bw + gap;
        if (x > end_x) end_x = x;
        continue;
      }
      node *n = ops[i];
      string label = get_node_label(n->type, n->value);
      int bw = text_w(label, scale) + pad * 2;
      int color_idx = 1;
      if (n->hash != 0 && hh == n->hash)
        color_idx = 4;
      draw_boxed_text(context, font, label, x, y, scale, bw, box_h, color_idx);
      app_state.node_zones.push_back({x, y, bw, box_h, n->hash});
      x += bw + gap;
      if (x > end_x) end_x = x;
    }
    return end_x;
  };

  // BEST program pipeline
  {
    int lw = text_w("BEST", scale) + pad * 2;
    int box_h = 5 * scale + pad * 2;
    draw_boxed_text(context, font, "BEST", lay.base_margin, lay.best_y, scale, lw, box_h, 5);
    if (task.best_tree)
      draw_pipeline(task.best_tree, lay.base_margin + lw + pad, lay.best_y);
  }

  // Elapsed timer
  {
    string elapsed_val = format_elapsed(elapsed);
    int lw = text_w("ELAPSED", scale) + pad * 2;
    draw_boxed_text(context, font, "ELAPSED", lay.base_margin, lay.elapsed_y, scale, lw, 5 * scale + pad * 2, 5);
    draw_text(context, font, elapsed_val, lay.base_margin + lw + pad, lay.elapsed_y + pad, scale);
  }
}

inline void update_display_buffer(grid &g, const grid &s, vector<int> &gb);

// Draws the fullscreen tree view with color legend
// Walk a tree (unary DSL) looking for a node whose hash matches.
inline node *find_node_by_hash(node *n, uint64_t h) {
  for (; n; n = n->arg)
    if (n->hash == h) return n;
  return nullptr;
}

// Where one grid lands on screen. One home for the placement maths, so an
// overlay drawn over the grids sits exactly where the grids do.
struct grid_place {
  int x = 0, y = 0, scale = 0;
  bool fits = false;
};

inline grid_place place_grid_cell(const grid &m, int col_idx, int row_y,
                                  const frame_layout &lay, int col_w, int slot_h) {
  grid_place p;
  if (m.width <= 0 || m.height <= 0) return p;
  int pad = lay.base_margin / 2;
  int outline = lay.scale;
  // The dimension label sits to the right of the grid, so it eats into the
  // width the grid may scale into.
  int dim_w = text_w("00x00", lay.scale) + outline * 2;
  p.scale = min((col_w - pad * 2 - dim_w - outline * 2) / m.width,
                (slot_h - pad * 2 - outline * 2) / m.height);
  if (p.scale < 1) return p;
  p.x = col_idx * col_w + lay.base_margin + outline;
  p.y = row_y + pad + outline;
  p.fits = true;
  return p;
}

// Draws the grid view (INPUT / OUTPUT), a quarter of the screen to a column.
// Returns false if there is no data to draw.
inline bool draw_grid_view(
    vulkan_context *context,
    const map<uint32_t, display_grid> &font,
    const display_grid &background,
    const puzzle &task,
    const puzzle_views &views,
    const frame_layout &lay) {

  if (task.train.empty() && task.test.empty())
    return false;

  int inputs_n = (int)(task.train.size() + task.test.size());
  int available_grid_h = lay.best_y - lay.grid_data_start_y;

  int n = inputs_n > 0 ? inputs_n : 1;
  int slot_h = available_grid_h / n;
  int col_w = lay.screen_w / 4;
  int cur_y = lay.grid_data_start_y;

  // The cells are the puzzle's own pair, and `at` says where they were
  // uploaded.
  auto draw_cell = [&](const grid &m, placement at, int col_idx, int row_y) {
    if (m.width <= 0 || m.height <= 0)
      return;

    int outline = lay.scale;
    string dim = to_text("%02dx%02d", m.width, m.height);
    grid_place p = place_grid_cell(m, col_idx, row_y, lay, col_w, slot_h);
    if (!p.fits) return; // grid band too small (e.g. footers ate the space) — skip
    int scale = p.scale;
    int grid_w = m.width * scale, grid_h = m.height * scale;
    int x = p.x, y = p.y;

    context->draw(background.at.offset, grid_w + outline * 2, grid_h + outline * 2,
                  x - outline, y - outline, 1, background.at.stride);
    context->draw(at.offset, m.width, m.height, x, y, scale, at.stride);

    int info_x = x + grid_w + outline * 2;
    draw_text(context, font, dim, info_x, y, lay.scale);
  };

  // WHETHER THE ANSWER IS SHOWN is not a thing anyone stores: a training pair
  // shows its output because that is what training means, and a test pair shows
  // its output once the puzzle is solved and there is nothing left to spoil.
  auto draw_section = [&](const vector<example> &vec, bool is_test) {
    if (vec.empty())
      return;
    const bool show_output = !is_test || task.is_solved;
    for (size_t i = 0; i < vec.size(); ++i) {
      const example &ex = vec[i];
      const example_views *v = views.of(is_test, i);
      // No view means the cells were never uploaded, so there is no placement to
      // draw from -- and a placement of nothing is stride 0, which would read the
      // start of the buffer over and over rather than draw an empty row.
      if (!v) {
        cur_y += slot_h;
        continue;
      }
      draw_cell(ex.input, v->input, 0, cur_y);
      if (show_output)
        draw_cell(ex.output, v->output, 1, cur_y);
      cur_y += slot_h;
    }
  };

  string heads[] = {"INPUT", "OUTPUT"};
  int grid_pad = 2 * lay.scale;
  for (int i = 0; i < (int)size(heads); ++i) {
    int header_label_x = i * col_w + lay.base_margin;
    int lw = text_w(heads[i], lay.scale) + grid_pad * 2;
    int lh = 5 * lay.scale + grid_pad * 2;
    draw_boxed_text(context, font, heads[i], header_label_x, lay.grid_labels_y, lay.scale, lw, lh, 5);
  }

  draw_section(task.train, false);
  draw_section(task.test, true);
  return true;
}

// Op palette overlay (toggle 'O'): a flow-laid list of every DSL op, each shown
// as Name or Name[0-N] for annotated ops. Read-only for now â€” the first piece of
// the in-GUI composer (next step: click a token to append it to a draft program
// and let the existing hover-preview engine show the effect live).
// Which tab an op is drawn under is its role, read straight off the op list
// (role_of, 8_vocabulary.hpp). See notes/2 - DSL/Op Algebra.md.

// Top-edge drawer overlay: a full-width panel that floats on top of the live
// frame, covering the header band from the top of the screen down to the
// grid-label row (no shrink â€” the grids and the BEST/ELAPSED footers stay
// visible below). A tab row picks the op family; the active family's ops
// flow below in as many columns as fit. Opens by dragging its handle, no key;
// tabs are click_zones.
// Thickness of the drawer/handle border (quad-weight).
inline int drawer_border_t(int scale) { return max(1, scale * 2); }

// Draw a white border rectangle around a region using the line sheet. stride=1
// so wide/tall edges can't overrun the solid sheet.
inline void draw_border(vulkan_context *context, int x, int y, int w, int h, int t,
                        int color = UI_COLOR_LINE) {
  if (!solid_sheets.count(color)) return;
  int off = solid_sheets[color].at.offset;
  context->draw(off, w, t, x, y, 1, 1);            // top
  context->draw(off, w, t, x, y + h - t, 1, 1);    // bottom
  context->draw(off, t, h, x, y, 1, 1);            // left
  context->draw(off, t, h, x + w - t, y, 1, 1);    // right
}

// The drawer's pull-handle. It rides the drawer's *bottom edge*: closed, that
// edge is at the top of the screen (handle hangs just below y=0); open, it sits
// at the drawer's current (draggable) bottom. Grab it and drag up/down to resize
// or close (drag-only â€” no click toggle). Centered; publishes its rect so the
// message thread can drive the drag.
inline void draw_op_handle(vulkan_context *context, global_app_state &app_state,
                           const frame_layout &lay, int drawer_bottom) {
  int scale = lay.scale;
  int steps = 4;                            // stacked rects forming the wedge
  int step_h = scale;                       // each step's height
  int box_h = steps * step_h;               // short overall
  int hw = lay.screen_w / 6;                // wide top, no outline
  int hx = (lay.screen_w - hw) / 2;
  int hy = drawer_bottom;                   // hangs off the drawer's bottom edge
  // Dim grey (11) idle -> bright grey (5) when the cursor is near, cheap "fade".
  // Pulled out, it stays bright: it is the only way back, so it must not fade
  // into the drawer it is holding open.
  int hcol = (app_state.op_handle_hot.load() || app_state.show_op_list.load()) ? 5 : 11;
  if (solid_sheets.count(hcol)) {
    int off = solid_sheets[hcol].at.offset;
    // Stacked rects, each ~step_h narrower per side -> shallow ~45deg wedge.
    for (int s = 0; s < steps; ++s) {
      int inset = s * step_h;
      int rw = hw - 2 * inset;
      if (rw <= 0) break;
      context->draw(off, rw, step_h, hx + inset, hy + s * step_h, 1, 1);
    }
  }
  // No click_zone: the handle opens/closes on DRAG only (a plain click does
  // nothing â€” no auto kick-down), and no key does either.
  // Publish handle rect so the message thread can start/track a drag from here.
  app_state.op_handle_x.store(hx);
  app_state.op_handle_y.store(hy);
  app_state.op_handle_w.store(hw);
  app_state.op_handle_h.store(box_h);
}

// A straight line between two points, any angle. The sheets only draw
// rectangles, so walk the longer axis and stamp a square every t pixels.
inline void draw_line(vulkan_context *context, int color,
                      int x0, int y0, int x1, int y1, int t) {
  if (!solid_sheets.count(color)) return;
  int off = solid_sheets[color].at.offset;
  int dx = x1 - x0, dy = y1 - y0;
  int steps = max(abs(dx), abs(dy)) / max(1, t);
  for (int i = 0; i <= steps; ++i) {
    int x = steps ? x0 + dx * i / steps : x0;
    int y = steps ? y0 + dy * i / steps : y0;
    context->draw(off, t, t, x - t / 2, y - t / 2, 1, 1);
  }
}

// THE PIECES NO LINE ANSWERS — the only ones the LINE stepper cannot reach, and
// so the only ones its own stepper has to carry. Every other piece is an end of
// some line, and walking it twice would be two names for one thing. The two
// steppers together reach every piece exactly once, and REFUSED carries the
// matches that never became a line at all: the three miss nothing between them.
// NOTHING FOUND IT is the test, carried from where the cells were — not
// `first_line < 0`, which is credited by overlapping BOXES and so called every
// piece sitting in a ring's hole answered by whatever answered the ring.

// A STEPPER WANTS ONE LIST, THE FACTS ARE NESTED. So the walk is a list of
// PLACES in the facts — which pair, which side, which one — and never a copy of
// what is there. Four bytes a row instead of thirty, and a row cannot say less
// than the fact it points at, which a copy always eventually did.
// Built for the ONE reading on screen, not all eighty, so it is a fraction of
// the work the stored copy did on every load.
struct piece_place {
  int8_t pair = 0;
  bool from_input = true;
  int16_t which = 0;
};
struct row_place {
  int8_t pair = 0;
  int16_t which = 0;
};

// The piece a place names. Which list it is in is the place's own side, the
// same way an edge carries the side it was read from.
inline const correspondence_election::object_info& piece_in(
    const vector<correspondence_election::pair_info>& pairs, const piece_place& at) {
  return pairs[at.pair].objects_on(at.from_input)[at.which];
}

// WHERE ONE EDGE RUNS, in cells: the box the piece sits in, and the box it
// landed on. Worked out where it is drawn rather than stored beside the edge —
// it is a reading of the facts, not another fact.
// Hanging off the edge, the landing is the piece's OWN size: the window is
// whole even where the grid stops.
struct line_ends {
  int8_t from_x = 0, from_y = 0, from_w = 0, from_h = 0;
  int8_t to_x = 0, to_y = 0, to_w = 0, to_h = 0;
};
inline line_ends ends_of(const correspondence_election::pair_info& pair,
                         const correspondence_election::edge& edge) {
  const auto& piece = pair.source_of(edge)->summary;
  const auto* target = pair.target_of(edge);
  const auto& landed = target ? target->summary : piece;
  return { piece.col, piece.row, piece.width, piece.height, edge.col, edge.row, landed.width, landed.height };
}

// What the edge CLAIMS, which the drawing cannot say: found as itself, found
// wearing another color, or not found at all and only sitting on what it
// overlaps most. Said in the order the claim weakens.
enum : int8_t { line_exact, line_recolored, line_sits_on };
inline int8_t claim_of(const correspondence_election::edge& edge) {
  return !edge.same_shape ? line_sits_on : edge.exact ? line_exact : line_recolored;
}

// The pieces of one reading, both grids, IN THE ORDER THE SCREEN READS: the
// pair, then the input column before the output one, then down the grid. The
// cut is biggest-first, which is not a thing on screen — walked that way the
// box you are stepping bounced between the columns.
inline vector<piece_place> pieces_of(const vector<correspondence_election::pair_info>& pairs) {
  vector<piece_place> all;
  for (size_t pair = 0; pair < pairs.size(); ++pair)
    for (int side = 0; side < 2; ++side) {
      const auto& list = pairs[pair].objects_on(side == 0);
      for (size_t i = 0; i < list.size(); ++i)
        all.push_back({ (int8_t)pair, side == 0, (int16_t)i });
    }
  stable_sort(all.begin(), all.end(), [&](const piece_place& a, const piece_place& b) {
    const auto& one = piece_in(pairs, a);
    const auto& two = piece_in(pairs, b);
    return tuple{ a.pair, !a.from_input, one.summary.row, one.summary.col } <
           tuple{ b.pair, !b.from_input, two.summary.row, two.summary.col };
  });
  return all;
}

// EVERY SET OF COPIES THERE ACTUALLY IS. A set is elected PER GRID, so a set
// number on its own names one set on every grid at once — the same number
// meaning a different set on each. One step here is one set on one grid, which
// is the only reading of it where a count, a size and a place mean one thing.
struct motif_set {
  int8_t pair = 0;
  bool from_input = true;
  int16_t which = -1;
  bool operator==(const motif_set&) const = default;
};
inline vector<motif_set> motif_sets(const vector<correspondence_election::pair_info>& pairs,
                                    const vector<piece_place>& pieces) {
  vector<motif_set> sets;
  for (const auto& at : pieces) {
    const auto& pc = piece_in(pairs, at);
    if (pc.motif < 0) continue;
    const motif_set here{ at.pair, at.from_input, pc.motif };
    if (ranges::find(sets, here) == sets.end()) sets.push_back(here);
  }
  return sets;
}
inline bool piece_is_in(const correspondence_election::object_info& pc,
                        const piece_place& at, const motif_set& set) {
  return pc.motif >= 0 && pc.motif == set.which &&
         at.pair == set.pair && at.from_input == set.from_input;
}

// THE PIECES THAT STAND ALONE — the ones no piece on the other grid is. Not
// the ones nothing can be said about: deleted, inserted and covered are all
// answers, and each one is said under the stepper. What these lack is a
// partner, which is why the walk is named for that and not for silence.
inline vector<piece_place> pieces_standing_alone(
    const vector<correspondence_election::pair_info>& pairs,
    const vector<piece_place>& pieces) {
  vector<piece_place> alone;
  for (const auto& at : pieces)
    if (!piece_in(pairs, at).has_partner && piece_in(pairs, at).instance_of < 0) alone.push_back(at);
  return alone;
}

// The refused matches, read the same way round, off the piece each was for.
inline vector<row_place> refusals_of(const vector<correspondence_election::pair_info>& pairs) {
  vector<row_place> all;
  for (size_t pair = 0; pair < pairs.size(); ++pair)
    for (size_t i = 0; i < pairs[pair].discarded.size(); ++i)
      all.push_back({ (int8_t)pair, (int16_t)i });
  stable_sort(all.begin(), all.end(), [&](const row_place& a, const row_place& b) {
    const auto& one = pairs[a.pair].discarded[a.which].match;
    const auto& two = pairs[b.pair].discarded[b.which].match;
    const auto& first = pairs[a.pair].source_of(one)->summary;
    const auto& second = pairs[b.pair].source_of(two)->summary;
    return tuple{ a.pair, !one.from_input, first.row, first.col } <
           tuple{ b.pair, !two.from_input, second.row, second.col };
  });
  return all;
}

// The cells the reading could not account for, in the order they were found,
// less the ones a fired rule accounts for after all: each place once, however
// many words it still waits on.
inline vector<row_place> left_over_of(const vector<correspondence_election::pair_info>& pairs) {
  vector<row_place> all;
  for (size_t pair = 0; pair < pairs.size(); ++pair) {
    bool taken[2][MAX_GRID_DIM][MAX_GRID_DIM] = {};
    for (size_t i = 0; i < pairs[pair].left_over.size(); ++i) {
      const auto& cell = pairs[pair].left_over[i];
      if (cell.claimed_by >= 0 || taken[cell.from_input][cell.row][cell.col]) continue;
      taken[cell.from_input][cell.row][cell.col] = true;
      all.push_back({ (int8_t)pair, (int16_t)i });
    }
  }
  return all;
}

// THE ONE READING ON SCREEN, walked ONCE A FRAME and handed to whoever draws
// it. Every walk below sorts, and the pane runs its body twice — to measure and
// to draw — so working them out where they are used cost three walks a frame.
struct reading_on_screen {
  const vector<correspondence_election::pair_info>* pairs = nullptr;
  vector<piece_place> pieces, alone;
  vector<row_place> refused, left_over, lines;
  vector<motif_set> sets;
};

// Every edge of the reading, both ways round, one list a stepper can walk.
inline vector<row_place> lines_of(const vector<correspondence_election::pair_info>& pairs) {
  vector<row_place> all;
  for (size_t pair = 0; pair < pairs.size(); ++pair)
    for (size_t i = 0; i < pairs[pair].edges.size(); ++i)
      all.push_back({ (int8_t)pair, (int16_t)i });
  return all;
}

// SAME SPOT, ANOTHER OUTLINE. Several overlays can each have something to say
// about one object, and drawn on the same rectangle the later one simply paints
// over the earlier — so only one of them was ever visible. Every outline after
// the first steps inside the last, in the order the sliders are listed: LINE
// outermost, then ALONE, REFUSED, MOTIF, LEFT OVER. It stops stepping when the box
// has no middle left to step into, which is what a one-cell object runs into.
// ONLY A DIFFERENT COLOR IS A DIFFERENT THING TO SAY. Many lines can land on
// one piece and each asks for the same white box; nesting those would grow one
// fat white border that reads as a heavier claim instead of a repeated one, so
// a color already drawn on a spot is dropped and never counted as a step.
struct nested_outlines {
  static constexpr int thickness = 2;
  struct spot {
    int x, y, w, h, color;
    bool operator==(const spot &) const = default;
    // TOUCHING THE SAME GROUND IS ENOUGH TO HAVE TO STEP. Asked as "the very
    // same rectangle", two boxes about one thing that were not the same size
    // never stepped: a landing that hangs off the wall is drawn at its whole
    // width while the clipped object under it is narrower, so they shared a
    // corner and their borders merged into one smear. Any shared ground means
    // one is drawn over the other, which is the thing stepping is for.
    bool shares_ground(const spot &o) const {
      return x < o.x + o.w && o.x < x + w && y < o.y + o.h && o.y < y + h;
    }
  };
  // Each ring remembers how far in it sat, so the next one can sit against it.
  struct ring { spot where; int step; };
  vector<ring> drawn;
  // `inside_mark` is a box that marks PART of a thing rather than ringing the
  // whole of it — the piece under a landing. It draws where it is and joins no
  // stack: counted as a ring it would push the next real ring a step further in
  // and leave a gap around nothing.
  void add(vulkan_context *context, int x, int y, int w, int h,
           int color = UI_COLOR_LINE, bool inside_mark = false) {
    const spot here{x, y, w, h, color};
    // ONE STEP INSIDE THE INNERMOST RING IT TOUCHES, never a step per ring
    // touched. Counting them all, two rings over the same spot plus anything
    // else overlapping put the third at depth two and left a black gap between
    // rings that are meant to read as one nest.
    int inner = -1;
    for (const ring &r : drawn) {
      if (r.where == here) return;
      if (r.where.shares_ground(here)) inner = max(inner, r.step);
    }
    const int room = (min(w, h) - thickness * 3) / (thickness * 2);
    const int step = inside_mark ? 0 : max(0, min(inner + 1, room));
    if (!inside_mark) drawn.push_back({ here, step });
    const int inset = step * thickness;
    draw_border(context, x + inset, y + inset, w - inset * 2, h - inset * 2,
                thickness, color);
  }
};

// Object correspondence overlay: one line per edge, center of the piece to
// center of where it landed, with a two-pixel box at each end. ONE DIRECTION
// PER READING — where each piece went — because reading it from the other end
// is the same fact, not a second one; WHICH end a line was read from is the
// LINE's own side, and it decides which column each box lands in.
// `rank` picks which reading to show ('C' steps it). `pick` is the one line to
// show alone ('S' walks them), or SHOW_ALL for all at once: lines landing on
// the same place hide each other, and one on screen cannot.
inline void draw_correspondence(vulkan_context *context, const puzzle &task,
                                const frame_layout &lay, int rank, int pick,
                                nested_outlines &rings) {
  int inputs_n = (int)(task.train.size() + task.test.size());
  const auto& correspondence = task.inf.correspondence;
  if (inputs_n <= 0 || rank >= (int)correspondence.readings.size()) return;
  const auto& pairs = correspondence.readings[rank].pairs;
  int slot_h = (lay.best_y - lay.grid_data_start_y) / inputs_n;
  int col_w = lay.screen_w / 4;
  int t = max(1, lay.scale);
  int nth = -1;
  for (const row_place& at : lines_of(pairs)) {
    if (pick >= 0 && ++nth != pick) continue;
    if (at.pair < 0 || at.pair >= (int)task.train.size()) continue;
    const auto& edge = pairs[at.pair].edges[at.which];
    const line_ends ends = ends_of(pairs[at.pair], edge);
    const example &ex = task.train[at.pair];
    int row_y = lay.grid_data_start_y + at.pair * slot_h;
    // Input is the left column, output the right; the line says which end it
    // starts on, so the two readings draw on the same grid without arguing.
    grid_place from = place_grid_cell(edge.from_input ? ex.input : ex.output,
                                      edge.from_input ? 0 : 1, row_y, lay, col_w, slot_h);
    grid_place to   = place_grid_cell(edge.from_input ? ex.output : ex.input,
                                      edge.from_input ? 1 : 0, row_y, lay, col_w, slot_h);
    if (!from.fits || !to.fits) continue;
    // Centers in HALF CELLS (2*x + w), so an even-sized box centers BETWEEN two
    // cells instead of being nudged off by one.
    draw_line(context, (int)arc_color::yellow,
              from.x + (2 * ends.from_x + ends.from_w) * from.scale / 2,
              from.y + (2 * ends.from_y + ends.from_h) * from.scale / 2,
              to.x + (2 * ends.to_x + ends.to_w) * to.scale / 2,
              to.y + (2 * ends.to_y + ends.to_h) * to.scale / 2, t);
    // The two boxes the line joins: where the piece sits, and where it landed.
    rings.add(context, from.x + ends.from_x * from.scale, from.y + ends.from_y * from.scale,
              ends.from_w * from.scale, ends.from_h * from.scale);
    // Hanging off the edge, the landing is the piece's own size, not the
    // clipped one — the window is whole even where the grid stops, and it is
    // drawn past the wall because that is what it says.
    const bool clipped = (edge.outside > 0);
    rings.add(context, to.x + ends.to_x * to.scale, to.y + ends.to_y * to.scale,
              (clipped ? ends.from_w : ends.to_w) * to.scale,
              (clipped ? ends.from_h : ends.to_h) * to.scale);
    // WHICH BIT OF IT WAS ALREADY THERE. A sits-on line says this piece is
    // under that one, and the box round the covering piece alone does not say
    // where underneath. So the piece's own outline is drawn a second time, on
    // the far grid, at its own place — a seed shows up inside the tile stamped
    // over it. Only for sits-on: every other line MOVED its piece, and drawing
    // it at its old place on the new grid would say it stayed.
    if (claim_of(edge) == line_sits_on)
      rings.add(context, to.x + ends.from_x * to.scale, to.y + ends.from_y * to.scale,
                ends.from_w * to.scale, ends.from_h * to.scale, UI_COLOR_LINE, true);
  }
}

// ONE SELECTED THING, on the grids. A box where it sits, and when something
// answers it a box there too with a line between — the same geometry the edges
// draw with, in another color, so a selection reads as a selection and not as
// one more edge. `starts_on_input` says which column the first box lives in.
// Without this a stepper that walks what the lines CANNOT show would itself
// show nothing, which is the wrong way round.
inline void draw_selection(vulkan_context *context, const puzzle &task,
                            const frame_layout &lay, int pair, bool starts_on_input,
                            int ax, int ay, int aw, int ah,
                            bool answered, int bx, int by, int bw, int bh, int color,
                            nested_outlines &rings) {
  int inputs_n = (int)(task.train.size() + task.test.size());
  if (inputs_n <= 0 || pair < 0 || pair >= (int)task.train.size()) return;
  int slot_h = (lay.best_y - lay.grid_data_start_y) / inputs_n;
  int col_w = lay.screen_w / 4;
  int t = max(1, lay.scale);
  const example &ex = task.train[pair];
  int row_y = lay.grid_data_start_y + pair * slot_h;
  grid_place a = place_grid_cell(starts_on_input ? ex.input : ex.output,
                                 starts_on_input ? 0 : 1, row_y, lay, col_w, slot_h);
  if (!a.fits) return;
  rings.add(context, a.x + ax * a.scale, a.y + ay * a.scale,
            aw * a.scale, ah * a.scale, color);
  if (!answered) return;
  grid_place b = place_grid_cell(starts_on_input ? ex.output : ex.input,
                                 starts_on_input ? 1 : 0, row_y, lay, col_w, slot_h);
  if (!b.fits) return;
  rings.add(context, b.x + bx * b.scale, b.y + by * b.scale,
            bw * b.scale, bh * b.scale, color);
  draw_line(context, color,
            a.x + (2 * ax + aw) * a.scale / 2, a.y + (2 * ay + ah) * a.scale / 2,
            b.x + (2 * bx + bw) * b.scale / 2, b.y + (2 * by + bh) * b.scale / 2, t);
}

// What inference read off this puzzle. EITHER/OR: on the FACTS step the pane
// reads out everything decided — size rule, colors, the canvas election, what
// changed, what correspondence made of each pair — and no lines are drawn. On a
// correspondence step the lines are on the grids and the pane says only what
// their colors mean, and the readout doubles as the controls: every row that
// names a choice is the thing you click to change it. Parked over the right
// half (cols 2-3).
inline void draw_inference_pane(vulkan_context *context, const map<uint32_t, display_grid> &font,
                                global_app_state &app_state,
                                const puzzle &task, const frame_layout &lay,
                                const reading_on_screen &walked,
                                int view, int rank, int cut, int cuts, int step, int drawn,
                                int piece_at, int dropped_at, int motif_at,
                                int left_over_at) {
  const bool facts_face = (view == global_app_state::INFERENCE_FACTS);
  // MEASURING PASS. The frame is drawn from what the body actually spends, so a
  // row added below is a row the frame already knows about. Counting the rows
  // twice -- once for the frame, once in the body -- is what let the text out.
  bool measuring = false;
  int scale = lay.scale;
  int pad = 2 * scale;
  int line_h = 7 * scale;
  int sw = 5 * scale;                 // color swatch side
  // Park it over the right half (cols 2-3).
  int col_w = lay.screen_w / 4;
  int x0 = 2 * col_w + lay.base_margin, y0 = lay.grid_data_start_y;
  int panel_w = 2 * col_w - lay.base_margin * 2;

  int t = max(1, scale / 2);          // swatch outline thickness
  auto swatch = [&](int color, int x, int y) {
    if (measuring) return;
    // white outline first, swatch inset on top â€” so black reads on black.
    draw_border(context, x, y, sw, sw, t);
    if (color >= 0 && color < COLOR_COUNT && solid_sheets.count(color))
      context->draw(solid_sheets[color].at.offset, sw - t * 2, sw - t * 2,
                    x + t, y + t, 1, solid_sheets[color].at.stride);
    else
      draw_text(context, font, "-", x + t, y + t, scale);
  };

  const auto& correspondence = task.inf.correspondence;
  const int readings = (int)correspondence.readings.size();

  auto& bg = task.inf.ground_election;
  auto& F = task.inf.colors;
  const int box_h = 5 * scale + pad * 2;   // a control row is a button, not a line
  // ONE ROW HEIGHT FOR THE WHOLE PANE, the control rows' own, so a list of text
  // reads at the same rhythm as a list of buttons. And ONE BLOCK GAP: the air
  // above a rule, below it, and between the frame and the first and last rows,
  // so nothing in the pane is spaced by a number chosen on its own.
  const int row_h = box_h + pad;
  // The air the pane keeps from its own frame, top and bottom.
  const int frame_pad = row_h;
  // The air above and below a sector rule. Smaller than the frame's, because a
  // break inside a list is not the same distance as the edge of the list.
  const int block_gap = row_h / 2;
  // ONE MARGIN, LEFT AND RIGHT ALIKE, and the sector rules run its full width —
  // so every row of text starts and ends exactly where the rules do.
  const int margin = pad * 9;
  // NOBODY COUNTS ROWS ANY MORE. Both faces run their body once to measure and
  // once to draw, so the frame is whatever the rows spent and a row added below
  // needs no arithmetic kept in step with it up here.

  int tx = x0 + margin, ty = y0 + (facts_face ? pad : frame_pad);

  // A NAME WEARS A CHIP, BOTH FACES ALIKE. Hoisted out of the controls body so
  // the facts rows can say a name the same way the steppers do.
  auto label_chip = [&](const string& name, int y, int x = -1) {
    if (x < 0) x = tx;
    if (!colored_fonts.count(0)) { draw_text(context, font, name, x, y, scale); return; }
    if (!measuring)
      fill_rect(context, UI_COLOR_LINE, x - pad / 2, y - pad / 2,
                text_w(name, scale) + pad, 5 * scale + pad);
    if (!measuring) draw_text(context, colored_fonts[0], name, x, y, scale);
  };
  // A ROW OF NAMED THINGS: each name in its chip, its value beside it, the next
  // name a gap further on. One row however many things it names.
  auto named_row = [&](std::initializer_list<std::pair<const char*, string>> parts) {
    int x = tx;
    for (const auto& part : parts) {
      label_chip(part.first, ty, x);
      x += text_w(part.first, scale) + pad * 2;
      if (!part.second.empty()) {
        if (!measuring) draw_text(context, font, part.second, x, ty, scale);
        x += text_w(part.second, scale) + pad * 4;
      }
    }
    ty += line_h;
  };
  const int right_x = x0 + panel_w - margin;
  // A RULE ACROSS THE PANE AND AIR EITHER SIDE OF IT. No title: a block that
  // needs a heading to be understood has under-named its own rows, and the
  // heading then says twice what the rows should have said once. The break
  // itself is all the sectioning needed. Spacing lives here alone, so every
  // block is parted from the next by the same gap whatever it holds.
  auto divide = [&]() {
    // EQUAL AIR EITHER SIDE, WHICH IS NOT THE SAME AS EQUAL NUMBERS HERE.
    // A row leaves `ty` a whole line below its own text, so that trailing
    // line is already part of the gap above the rule — matching the two
    // numbers made the gap above a line and a half bigger than the gap below.
    // Below therefore gets one line more than above, and they draw even.
    ty += block_gap;
    if (right_x > tx)
      fill_rect(context, UI_COLOR_LINE, tx, ty, right_x - tx, max(1, scale / 2));
    ty += block_gap + row_h - 5 * scale;
  };

  auto line = [&](const string& s) {
    if (!measuring) draw_text(context, font, s, tx, ty, scale);
    ty += line_h;
  };

  // A boxed word the mouse can press, in the header buttons' clothes. Hoisted
  // out of the controls body so the facts face can fold its programs with one.
  auto button = [&](const string& label, int x, int y, ui_action act) {
    int bw = text_w(label, scale) + pad * 2;
    draw_boxed_text(context, font, label, x, y, scale, bw, box_h,
                    app_state.active_ui_action == act ? 4 : 1);
    if (!measuring) app_state.click_zones.push_back({x, y, bw, box_h, act});
    return bw;
  };

  if (!facts_face) {
    // ONE BODY, RUN TWICE: measured first, drawn second, so the frame is sized
    // by what the rows actually spend and adding one needs no bookkeeping.
    auto paint_the_controls = [&]() {
      // Every control row reads NAME, what it is now, then the control. The name
      // column and the value field are fixed widths that don't depend on what they
      // say, so every control begins at the same place and none of them moves. The
      // value field holds only "WHICH OF HOW MANY", the widest of which is a track
      // saying OFF or ALL of some hundreds. What the segmentation IS gets a line of its
      // own below, because saying it properly takes a sentence and a sentence
      // would push every control off the panel.
      const int value_chars = (int)string("OFF OF 9999").size();
      // EVERY LABEL THE PANE USES, so the value column is as wide as the widest
      // of them. This was the width of "DROPPED" alone, and the moment a label
      // grew past it the name ran straight through its own value.
      static const char* const pane_labels[] = {
          "CUT", "LINE", "LEFT OVER", "REFUSED", "MOTIF" };
      int label_col = 0;
      for (const char* name : pane_labels) label_col = max(label_col, text_w(name, scale));
      // THREE CHARACTERS CLEAR OF THE LONGEST LABEL. At one character the widest
      // of them sat right against its own value and read as one word.
      const int value_x = tx + label_col + pad * 6;
      const int control_x = value_x + text_w(string(value_chars, ' '), scale) + pad * 6;
      // `chosen` inverts the value — black on white — for the reading inference
      // actually decided on. Two readings can score the same to the last cell, so
      // which one leads has to be visible, not remembered from an ordering.
      auto row_head = [&](const string& name, const string& value, bool chosen = false) {
        label_chip(name, ty + pad);
        if (chosen && colored_fonts.count(0)) {
          fill_rect(context, UI_COLOR_LINE, value_x - pad / 2, ty + pad / 2,
                    text_w(value, scale) + pad, 5 * scale + pad);
          draw_text(context, colored_fonts[0], value, value_x, ty + pad, scale);
          return;
        }
        draw_text(context, font, value, value_x, ty + pad, scale);
      };
      // A named line of plain text on the same column the controls use, so every
      // label in the pane lines its value up on one edge.
      auto named_line = [&](const string& name, const string& value) {
        label_chip(name, ty);
        draw_text(context, font, value, value_x, ty, scale);
        ty += row_h;
      };
      // A track with a handle on the stop it's on, filling the rest of the row.
      // Press or drag anywhere along it to land there. The handle is one size
      // whatever the puzzle offers — only where it sits says anything — and it
      // wears the color of what it walks, the same color that thing is drawn in on
      // the grids. It travels the track less its own width, so the first
      // and last stops park it flush inside either end and both are plain to see.
      auto slider = [&](int x, int stop, int stops, int color, track* which) {
        const int w = right_x - x;
        const int cap = max(2, scale * 2);
        const int knob_w = min(6 * scale, max(1, w - cap * 2));
        if (w <= cap * 2 || stops <= 0) return;
        // THE HANDLE RUNS BETWEEN THE END STOPS, NOT UNDER THEM. Drawing the caps
        // over it hid the handle at either extreme, which is exactly where you
        // most need to see where it sits. Its run is inset by a cap at each end,
        // and the press zone is inset with it so the two agree on where a stop is.
        const int run_x = x + cap, run_w = w - cap * 2;
        const int travel = max(1, run_w - knob_w);
        const int kx = run_x + (stops > 1 ? travel * stop / (stops - 1) : 0);
        fill_rect(context, 11, x, ty, w, box_h);
        fill_rect(context, UI_COLOR_LINE, x, ty, cap, box_h);
        fill_rect(context, UI_COLOR_LINE, x + w - cap, ty, cap, box_h);
        fill_rect(context, color, kx, ty, knob_w, box_h);
        app_state.slider_zones.push_back({run_x, ty, run_w, box_h, which, stops, knob_w});
      };


      // NO RULE ABOVE THE FIRST BLOCK — a divider parts two things, and there is
      // nothing above this one to part it from.
      // Which way of cutting the grid is on screen, and how many there are. A
      // cut is shown at the one reading it kept, the combination of freedoms that
      // entered the election for it; the rest are in the log.
      // COUNTED FROM ZERO, like the log and like the lists themselves. Counted
      // from one it read as a different number from the same thing, and every
      // sentence about a reading had to say which of the two it meant.
      row_head("CUT", to_text("%4d OF %d", cut, max(1, cuts)), rank == correspondence.chosen);
      {
        int x = control_x + button("< [SHIFT C]", control_x, ty, ui_action::cycle_coarseness_back) + pad;
        button("> [C]", x, ty, ui_action::cycle_coarseness);
      }
      ty += box_h + pad;
      // ONE SHAPE FOR ALL THREE, because they answer the same question about
      // three lists: none of it, one of them, all of it — OFF at the left, the
      // items in between, ALL at the right. The steps beside the track land on
      // the one next door; the track itself is how you cross a long list.
      auto track_row = [&](const string& name, const string& key, int at, int count,
                           int color, ui_action back, ui_action fwd, track* which) {
        // RIGHT-ALIGNED, because OFF is three characters and 7 is one, and a
        // wandering OF makes four rows read as four different shapes.
        // Counted from zero, the same as CUT above and the same as the log.
        row_head(name, at == SHOW_OFF ? to_text("%4s OF %d", "OFF", count)
                     : at == SHOW_ALL ? to_text("%4s OF %d", "ALL", count)
                                      : to_text("%4d OF %d", at, count));
        int x = control_x + button(to_text("< [SHIFT %s]", key.c_str()), control_x, ty, back) + pad;
        x += button(to_text("> [%s]", key.c_str()), x, ty, fwd) + pad;
        slider(x, track::to_stop(at, count), count + 2, color, which);
        ty += box_h + pad;
      };
      // WHAT THE STEPPER IS ON, SAID DIRECTLY UNDER IT. Two blocks apart, each of
      // these had to repeat its own label to say which stepper it belonged to;
      // under it, the stepper's name is the label. ALWAYS A ROW HIGH, said or not,
      // so nothing below shifts as you step.
      // A LINE OF TEXT IS SHORTER THAN A ROW OF CONTROLS, and taking a control
      // row each is what ran the panel off the bottom of a 1080p screen.
      // HARD LEFT, not lined up with the values above it. These are sentences,
      // not values, and the longest of them ran off the panel when it started
      // where a value starts. Nothing sits to their left to line up with anyway.
      auto detail = [&](const string& value) {
        if (!value.empty()) draw_text(context, font, value, tx, ty, scale);
        ty += line_h;
      };
      if (rank >= readings) return;
      const auto& R = correspondence.readings[rank];
      const auto& pairs = correspondence.readings[rank].pairs;
      // WHICH CUT THIS IS, in words, and the reading's number, which is what the
      // log calls it. What its pieces were allowed to do is the list below:
      // what they actually did.
      detail(to_text("%s %s  ---  READING %d", segmentation_names_lead,
                     segmentation_names[clamp((int)R.segmentation, 0,
                                       (int)segmentation_methods::count)],
                     rank));
      static const char* kind_name[] = { "EXACT", "RECOLORED", "SITS ON" };
      // The pieces the reading cut out, and the matches it threw away. Neither is
      // reachable through the lines: a piece with no edge draws nothing, and a
      // thrown-away match was never drawn at all.
      const auto& pieces = walked.pieces;
      const auto& alone = walked.alone;
      const auto& refused = walked.refused;
      const auto& left_over = walked.left_over;
      const auto& lines = walked.lines;
      const int pieces_n = (int)alone.size();
      const int dropped_n = (int)refused.size();
      const int left_over_n = (int)left_over.size();
      track_row("LINE", "S", step, drawn, (int)arc_color::yellow,
                ui_action::step_edge_back, ui_action::step_edge, &app_state.inference_edge);
      // THE ONE LINE YOU ARE ON, said in full. All of this is invisible in the
      // drawing: a line looks the same whether the piece was found as itself,
      // found in another color, or never found at all.
      {
        string said;
        if (step >= 0 && step < (int)lines.size()) {
          const row_place& at = lines[step];
          const auto& edge = pairs[at.pair].edges[at.which];
          const line_ends ends = ends_of(pairs[at.pair], edge);
          const int8_t color = pairs[at.pair].source_of(edge)->color;
          said = to_text("PAIR %d  COLOR %s  %dX%d AT %d,%d -> %d,%d  %s",
                         (int)at.pair, (color < 0 ? string("MIXED") : to_string(color)).c_str(),
                         (int)ends.from_w, (int)ends.from_h, (int)ends.from_x, (int)ends.from_y,
                         (int)ends.to_x, (int)ends.to_y, kind_name[claim_of(edge)]);
          if (edge.pose & 1) said += ", TURNED";
          if (edge.pose & 2) said += ", FLIPPED ROWS";
          if (edge.pose & 4) said += ", FLIPPED COLS";
          if (edge.hidden)   said += to_text(", %d HIDDEN", (int)edge.hidden);
        }
        detail(said);
      }
      track_row("REFUSED", "X", dropped_at, dropped_n, (int)arc_color::red,
                ui_action::step_dropped_back, ui_action::step_dropped, &app_state.inference_dropped);
      // THE ONE THROWN-AWAY MATCH YOU ARE ON: the piece, where it wanted to land,
      // what it was claiming, and why it lost.
      {
        string said;
        if (dropped_at >= 0 && dropped_at < dropped_n) {
          const row_place& at = refused[dropped_at];
          const auto& [dr, reason] = pairs[at.pair].discarded[at.which];
          said = to_text("PAIR %d  PIECE %d  WANTED %d,%d  %s  %s",
                         (int)at.pair, (int)dr.from, (int)dr.col, (int)dr.row,
                         kind_name[claim_of(dr)],
                         correspondence_election::refusal_names[
                             clamp((int)reason, 0, 3)]);
          if (dr.hidden) said += to_text(", %d HIDDEN", (int)dr.hidden);
        }
        detail(said);
      }
      const auto& sets = walked.sets;
      track_row("MOTIF", "O", motif_at, (int)sets.size(), (int)arc_color::magenta,
                ui_action::step_motif_back, ui_action::step_motif, &app_state.inference_motif);
      // THE SET OF COPIES YOU ARE ON — one set, on one grid, so its count, its
      // size and where its leader sits all mean one thing. The leader is the
      // medoid: the member that disagrees least with all the others, and the one
      // drawn magenta while the rest are blue.
      {
        string said;
        if (motif_at >= 0 && motif_at < (int)sets.size()) {
          const motif_set& set = sets[motif_at];
          int members = 0, leader_w = 0, leader_h = 0, leader_x = 0, leader_y = 0;
          for (const piece_place& at : pieces) {
            const auto& pc = piece_in(pairs, at);
            if (!piece_is_in(pc, at, set)) continue;
            ++members;
            if (!pc.medoid) continue;
            leader_w = pc.summary.width; leader_h = pc.summary.height;
            leader_x = pc.summary.col;  leader_y = pc.summary.row;
          }
          said = to_text("PAIR %d %s, %d COPIES OF %dX%d, LEADER AT %d,%d",
                         (int)set.pair, set.from_input ? "INPUT" : "OUTPUT",
                         members, leader_w, leader_h, leader_x, leader_y);
        }
        detail(said);
      }
      // LAST OF THE STEPPERS, because its readout is the only one that needs two
      // lines and the pair of them sit better against the rule below than wedged
      // between two other controls.
      track_row("ALONE", "P", piece_at, pieces_n, (int)arc_color::gray,
                ui_action::step_piece_back, ui_action::step_piece, &app_state.inference_piece);
      // WHICH KIND OF ALONE IT IS. Deleted, inserted and covered are ANSWERS, so
      // the walk is not named for their absence — it is named for the one thing
      // they share, having no piece on the other grid that is them.
      // NOTHING FOUND IT is one fact, not two: which grid the
      // piece sits on is a property of the piece, and swapping which side is
      // called input turns either word into the other. So it is said FROM THE
      // INPUT SIDE and says so in the same breath, because a bare INSERTED leaves
      // the reader to remember whose standpoint it was, which is the thing nobody
      // remembers a day later. A piece with an edge but no line is neither:
      // something found a place for it and never found IT.
      // TWO LINES: the piece, then the patch of background it covers on the other
      // grid. On one line the pair of them ran past the panel, and the widest
      // screen is the tighter of the two.
      {
        string piece_said, blank_said;
        if (piece_at >= 0 && piece_at < pieces_n) {
          const piece_place& at = alone[piece_at];
          const auto& pc = piece_in(pairs, at);
          const char* silence = pc.edges_touching > 0 ? "ONLY SITS ON SOMETHING"
                              : at.from_input ? "DELETED, IT IS NOT IN THE OUTPUT"
                                              : "INSERTED, IT WAS NOT IN THE INPUT";
          piece_said = to_text("%s  PAIR %d  COLOR %s  %dX%d AT %d,%d  %d CELLS",
                               silence, (int)at.pair,
                               (pc.color < 0 ? string("MIXED") : to_string(pc.color)).c_str(),
                               (int)pc.summary.width, (int)pc.summary.height, (int)pc.summary.col, (int)pc.summary.row,
                               (int)pc.summary.count);
          if (pc.blank.found())
            blank_said = to_text("%s %dX%d AT %d,%d  %d CELLS",
                                 pc.blank.enclosed ? "FILLS A HOLE"
                                                   : "OVER OPEN SPACE",
                                 (int)pc.blank.box.width, (int)pc.blank.box.height, (int)pc.blank.box.col,
                                 (int)pc.blank.box.row, (int)pc.blank.box.count);
        }
        detail(piece_said);
        detail(blank_said);
      }
      // A CELL, NOT A PIECE. Everything above walks things the reading made —
      // pieces, matches, sets — and a cell it could not account for is none of
      // them, so it was the one fact on the pane with nothing behind it to point
      // at. The number here IS this list's length, so they cannot disagree.
      track_row("LEFT OVER", "U", left_over_at, left_over_n, (int)arc_color::orange,
                ui_action::step_left_over_back, ui_action::step_left_over,
                &app_state.inference_left_over);
      {
        string said;
        if (left_over_at >= 0 && left_over_at < left_over_n) {
          const row_place& at = left_over[left_over_at];
          const auto& cell = pairs[at.pair].left_over[at.which];
          // MADE, GONE, or simply not accounted for — the mirror this pane used
          // to read dropped that field, so the number could be said and the
          // reason could not. A cell a change carried names every word it still
          // waits on instead.
          string why;
          for (const auto& same : pairs[at.pair].left_over) {
            if (same.claimed_by >= 0 || same.from_input != cell.from_input ||
                same.row != cell.row || same.col != cell.col) continue;
            if (!why.empty()) why += ", ";
            why += same.waits >= 0 ? "NO WORD FOR " + string(correspondence_election::missing_words[same.waits])
                 : !same.made_or_gone ? "UNEXPLAINED"
                 : same.from_input    ? "IT WENT NOWHERE"
                                      : "IT WAS MADE HERE";
          }
          said = to_text("PAIR %d  %s  CELL AT %d,%d  %s", (int)at.pair,
                         cell.from_input ? "INPUT" : "OUTPUT", (int)cell.col, (int)cell.row, why.c_str());
        }
        detail(said);
      }
      // A CONTROL ROW IS TALLER THAN A TEXT ROW, so it leaves more behind it than
      // divide() expects, and the first rule sat lower in its gap than the rest.
      // Give back the difference and all the gaps draw the same.

      // WHAT THE CUT ON SCREEN WAS TAKEN TO SAY, a rule a row, each with what it
      // accounts for: the input pieces its edges are about and the cells on both
      // grids they carry. A rule that reads another's evidence sits a step under
      // it, and says what it names instead of counting cells, since the cells
      // were already the other rule's. The list is as long as what fired, so a
      // reading that needed three things shows three and one that needed none
      // shows none. The last row is what all of them together leave over: the
      // rules composed when it reads zero.
      divide();
      {
        const auto order = R.fired_in_order();
        // THE NAMES ARE ONE COLUMN however deep they sit, so every count
        // starts on one edge and a sum can be read straight down.
        const int step_in = pad * 6;
        int name_w = text_w("LEFT OVER", scale);
        for (const auto& [which, depth] : order)
          name_w = max(name_w, depth * step_in + text_w(R.fired[which].name, scale));
        const int said_x = tx + name_w + pad * 6;
        for (const auto& [which, depth] : order) {
          const auto& rule = R.fired[which];
          label_chip(rule.name, ty, tx + depth * step_in);
          // Counts only for a rule that accounts for something, but always the
          // same width, so what a rule says starts on one edge down the list.
          // A rule that stands alone explains every pair by itself, not a share.
          const string counts = rule.pieces || rule.cells
              ? to_text("%3d %-6s %4d CELLS", rule.pieces, rule.pieces == 1 ? "PIECE" : "PIECES", rule.cells)
              : to_text("%-21s", rule.stands_alone ? "BY ITSELF" : "");
          string said = counts;
          if (!rule.says.empty()) said += "  " + rule.says;
          if (!rule.no_word_for.empty()) said += "  NO WORD FOR " + rule.no_word_for;
          if (!measuring) draw_text(context, font, said, said_x, ty, scale);
          ty += row_h;
        }
        label_chip("LEFT OVER", ty);
        if (!measuring)
          draw_text(context, font, to_text("%11s%4d CELLS OF %d ON BOTH GRIDS", "", R.unclaimed(),
                                           (int)correspondence.cells), said_x, ty, scale);
        ty += row_h;
      }

      // The same questions asked of every line of the reading, pair by pair, plus
      // how the pieces were settled and what was thrown out on the way. CLASS is
      // the two extremes read off the pieces of THIS reading: the most pieces that
      // met at one landing, and the most places one piece went. So 1-1 is a plain
      // swap, 1-9 is one piece stamped nine times, 2-1 is two meeting at one.
      divide();
      static const char* how_name[] = { "FIT", "GREEDY", "GAVE UP" };
      // ONE ARRAY OF COLUMNS, so the heads and the rows cannot drift apart and
      // EACH HEAD IS ITS OWN CHIP — one chip over the whole line fused ten labels
      // into a single white bar with nothing between them.
      static const struct { const char* name; int width; } pair_columns[] = {
          {"PAIR", 4}, {"SETTLED", 7}, {"CLASS", 6}, {"EXACT", 6},
          {"RECOLORED", 10}, {"SITS ON", 8}, {"HIDDEN", 7}, {"TURNED", 7},
          {"REFUSED", 8}, {"OVER", 5} };
      const int char_w = 4 * scale;
      // A cell is right-aligned in its field, so its left edge is the field's end
      // less its own length — the same sum for a head and for a number below it.
      auto pair_cell_x = [&](int column, int text_chars) {
          int at = 0;
          for (int i = 0; i < column; ++i) at += pair_columns[i].width + 1;
          return tx + (at + pair_columns[column].width - text_chars) * char_w;
      };
      auto pair_row = [&](auto&&... cells) {
          int column = 0;
          for (const string& cell : { string(cells)... }) {
              draw_text(context, font, cell, pair_cell_x(column, (int)cell.size()),
                        ty, scale);
              ++column;
          }
          ty += row_h;
      };
      for (int i = 0; i < (int)size(pair_columns); ++i)
          label_chip(pair_columns[i].name, ty,
                     pair_cell_x(i, (int)string(pair_columns[i].name).size()));
      ty += row_h;
      for (size_t i = 0; i < pairs.size(); ++i) {
        const auto& pr = pairs[i];
        int kinds[3] = {}, hidden = 0, turned = 0;
        for (const auto& edge : pr.edges) {
          kinds[claim_of(edge)]++;
          hidden += (edge.hidden > 0);
          turned += (edge.pose != 0);
        }
        // Places one piece went is an input-side question, pieces meeting at one
        // landing an output-side one. Both ends are read, so both are counted.
        int most_landings = 0, most_sources = 0;
        for (const auto& piece : pr.input_info)
          most_landings = max(most_landings, piece.edges_touching);
        for (const auto& landing : pr.output_info)
          most_sources = max(most_sources, landing.edges_touching);
        pair_row(to_string(i), how_name[clamp((int)pr.verdict, 0, 2)],
                 to_text("%d-%d", most_sources, most_landings),
                 to_string(kinds[0]), to_string(kinds[1]), to_string(kinds[2]),
                 to_string(hidden), to_string(turned),
                 to_string(pr.discarded.size()), to_string(pr.unclaimed()));
      }
      // Why the dropped ones were dropped, over the whole reading. EVERY reason
      // there is: the copy this pane used to read had three buckets for four
      // reasons, so two of them were reported as one.
      {
        const int reasons = (int)size(correspondence_election::refusal_names);
        vector<int> by_reason(reasons, 0);
        for (const auto& pr : pairs)
          for (const auto& rejected : pr.discarded)
            if (rejected.reason >= 0 && rejected.reason < reasons) by_reason[rejected.reason]++;
        // ONE REASON A LINE. All three on one line came to more characters than
        // the panel is wide, and the widest screen is the tighter of the two:
        // the label column grows with the font while the panel grows with the
        // screen. The label is said once and the rest sit under it.
        for (int reason = 0; reason < reasons; ++reason) {
          if (reason == 0) label_chip("REFUSED", ty);
          draw_text(context, font,
                    to_text("%d %s", by_reason[reason],
                            correspondence_election::refusal_names[reason]),
                    value_x, ty, scale);
          ty += line_h;
        }
      }
    };

    const int ty_top = ty;
    measuring = true;  paint_the_controls();  measuring = false;
    const int measured_h = (ty - ty_top) + frame_pad * 2;
    ty = ty_top;
    fill_rect(context, 0, x0, y0, panel_w, measured_h);
    draw_border(context, x0, y0, panel_w, measured_h, max(1, scale));
    paint_the_controls();
    return;
  }

  // ONE BODY, RUN TWICE: once to find out how tall it is, once to draw it.
  auto paint_the_facts = [&]() {
    {
      // Size · what decides how big the output is.
      static const char* rule_name[] = { "NOT INFERRED", "SAME AS INPUT", "FIXED",
                                         "SCALED", "WHAT IT DRAWS" };
      string s = to_text("OUTPUT SIZE  %s", rule_name[(int)task.inf.size.Chosen]);
      if (task.inf.size.Chosen == output_dimension_rule::Fixed)
        s += to_text("  %dX%d", (int)task.inf.size.width, (int)task.inf.size.height);
      if (task.inf.size.Chosen == output_dimension_rule::Scale)
        s += to_text("  %d/%d BY %d/%d", (int)task.inf.size.width_numerator, (int)task.inf.size.width_denominator,
                     (int)task.inf.size.height_numerator, (int)task.inf.size.height_denominator);
      line(s);
    }
    named_row({{"COLORS GONE", color_facts::as_text(F.consumed)},
               {"NEW", color_facts::as_text(F.introduced)},
               {"KEPT", color_facts::as_text(F.preserved)}});
    named_row({{"IN EVERY INPUT", color_facts::as_text(F.in_every_input)},
               {"IN EVERY OUTPUT", color_facts::as_text(F.in_every_output)}});
    named_row({{"IN EVERY GRID", color_facts::as_text(F.in_every_grid)},
               {"ANYWHERE", color_facts::as_text(F.in_any_grid)}});
    {
      string all;
      for (int i = 0; i < task.inf.colors.output_colors_count; ++i)
        all += to_text("%d", (int)task.inf.colors.output_colors[i]);
      string by_size;
      for (int i = 0; i < task.inf.colors.output_colors_by_size_count; ++i)
        by_size += to_text("%d", (int)task.inf.colors.output_colors_by_size[i]);
      if (by_size.empty()) named_row({{"OUTPUT COLORS", all}});
      else named_row({{"OUTPUT COLORS", all}, {"BY SIZE", by_size}});
    }

    // BACKGROUND IS ITS OWN SECTOR: the elected canvases, then the ranked
    // hypotheses each election ran on. Ruled off at both ends, so the numbers
    // in the tables are never read as belonging to the color facts above them.
    divide();
    {
      if (bg.is_per_pair())
        named_row({{"BG INPUT", to_text("%d", (int)bg.input_elected())},
                   {"OUTPUT", to_text("%d", (int)bg.output_elected())},
                   {"MODE", "PER-PAIR"}});
      else
        named_row({{"BG INPUT", to_text("%d", (int)bg.input_elected())},
                   {"OUTPUT", to_text("%d", (int)bg.output_elected())}});
    }
    named_row({{"VETOED", bg.input_vetoed ? "YES" : "NO"}});
    {
      string pp;
      for (int8_t c : bg.across_input_output) pp += to_text(" %d", (int)c);
      named_row({{"PER PAIR", pp}});
      ty += line_h;  // blank line after
    }

    auto draw_dir = [&](const char* name, const vector<background_election::hypothesis>& d) {
      label_chip(name, ty); ty += line_h;
      // column header, aligned with the rows below
      // Every cell is text in a width, so heads and rows share one shape.
      static const char* bg_columns = "%7s %10s %10s %8s %7s %6s %6s %6s";
      draw_text(context, font, to_text(bg_columns,
                "COLOR", "EMBEDS", "DOMINANCE", "IN_OUT", "SCORE", "SPAN", "GAPS", "UNDER"),
                tx + sw + scale, ty, scale);
      ty += line_h;
      for (const auto& c : d) {
        swatch(c.color, tx, ty);
        string label = (c.color < 0) ? "NONE" : to_string((int)c.color);
        string row = to_text(bg_columns,
                             label.c_str(), to_string(c.embeds).c_str(), to_string(c.dominance).c_str(),
                             c.appears_in_any_output ? "YES" : "NO", to_string(c.score).c_str(),
                             to_string(c.span).c_str(), to_string(c.gaps).c_str(),
                             c.underneath ? "YES" : "NO");
        draw_text(context, font, row, tx + sw + scale, ty, scale);
        ty += line_h;
      }
      ty += line_h / 2;   // gap between elections
    };
    draw_dir("ACROSS INPUTS", bg.across_inputs);
    draw_dir("ACROSS OUTPUTS", bg.across_outputs);
    draw_dir("ACROSS IN+OUT", bg.across_inputs_outputs);
    divide();

    // What changed between the two sides.
    if (task.inf.unchanged.computed)
      line(to_text("INPUT ALREADY RIGHT ON %.0f PCT OF CELLS, WORST PAIR",
                   task.inf.unchanged.min_kept_fraction * 100.0f));
    if (task.inf.color_diff.computed) {
      string s = "CELLS WON/LOST ";
      for (int c = 0; c < COLOR_COUNT; ++c)
        if (task.inf.color_diff.gained_count[c] || task.inf.color_diff.lost_count[c])
          s += to_text(" %d:%d/%d", c, (int)task.inf.color_diff.gained_count[c], (int)task.inf.color_diff.lost_count[c]);
      line(s);
    }
    if (task.inf.color_swaps.count) {
      string s = "SWAP CANDIDATES";
      int k = 0;
      for (int a = 0; a < COLOR_COUNT; ++a)
        for (int b = a + 1; b < COLOR_COUNT; ++b, ++k)
          if (task.inf.color_swaps.variant[k]) s += to_text(" %d<>%d", a, b);
      line(s);
    }
    if (task.inf.shape_color_map.count)
      named_row({{"SHAPE PICKS COLOR", to_text("%d LEARNED", (int)task.inf.shape_color_map.count)}});

    // WHAT THE LOOKING SAID, IN THE SOLVER'S OWN WORDS. Shown as the programs
    // SPACE runs, not a prettier retelling, so what is on screen and what is
    // run cannot drift apart. One sentence a row, four beats to a sentence.
    // A LONG PROGRAM FOLDS: its first rows, then how many more it has. The
    // button on the heading opens every row and folds them again; every row at
    // once ran the pane off the bottom of the screen (31aa019c's fifteen
    // sentences, f823c43c's nineteen). On the heading, so it stays in reach
    // however far the open rows run.
    if (!task.suggested_programs.empty()) {
      constexpr size_t folded_rows = 6;
      const bool whole = app_state.show_whole_programs.load();
      bool folds = false;
      for (const string& program : task.suggested_programs)
        folds = folds || sentences_of(program).size() > folded_rows;
      if (folds) {
        const string fold = whole ? "FOLD" : "SHOW ALL";
        button(fold, right_x - (text_w(fold, scale) + pad * 2), ty - pad, ui_action::toggle_programs);
      }
      named_row({{"WHAT THE READING SAYS",
                  to_text("%d PROGRAM%s", (int)task.suggested_programs.size(),
                          task.suggested_programs.size() == 1 ? "" : "S")}});
      if (folds) ty += row_h - line_h;
      // Folded, a row too wide for the pane ends in "..." at the last word that
      // fits inside the frame; open, every row is said whole.
      auto fitted = [&](string row) {
        const int room = right_x - tx;
        if (whole || text_w(row, scale) <= room) return row;
        while (!row.empty() && text_w(row + " ...", scale) > room) row.pop_back();
        if (const size_t space = row.rfind(' '); space != string::npos) row.resize(space);
        return row + " ...";
      };
      for (size_t which = 0; which < task.suggested_programs.size(); ++which) {
        const vector<string> rows = sentences_of(task.suggested_programs[which]);
        const size_t shown = whole ? rows.size() : min(rows.size(), folded_rows);
        for (size_t row = 0; row < shown; ++row)
          line(fitted(to_text("  %s %s", row == 0 ? to_text("%d", (int)which + 1).c_str() : " ", rows[row].c_str())));
        if (shown < rows.size()) line(to_text("    ... %d MORE", (int)(rows.size() - shown)));
      }
    }

    // What correspondence made of each pair, said FROM THE INPUT'S SIDE, which is
    // the same standpoint the ALONE row uses. DELETED and INSERTED are one fact
    // read from two ends — no piece on the other grid is this one — and which end
    // is just which grid it sits on, so the two must not read as two different
    // findings. CLASS is the two extremes the six counts hide: most pieces
    // meeting at one landing, most places one piece went. 1-1 is a plain swap,
    // 1-9 is one piece stamped nine times, 3-2 is both at once.
    ty += line_h / 2;
    // Every cell is text in a width, so the head and the rows share one shape.
    const char* pair_columns_wide =
        "%4s %9s %5s %6s %9s %8s %6s %8s %8s %5s";
    draw_text(context, font,
              to_text("%4s %s %s %8s %6s",
                      "", align_centre("EACH INPUT PIECE", 22).c_str(),
                      align_centre("EACH OUTPUT PIECE", 25).c_str(), "", ""),
              tx, ty, scale);
    ty += line_h;
    draw_text(context, font,
              to_text(pair_columns_wide,
                      "PAIR", "DELETED", "KEPT", "COPIED",
                      "INSERTED", "FROM ONE", "MERGED", "CLASS", "CARRIED", "MADE"),
              tx, ty, scale);
    ty += line_h;
    for (size_t i = 0; i < correspondence.chosen_pairs().size(); ++i) {
      const auto& p = correspondence.chosen_pairs()[i];
      line(to_text(pair_columns_wide,
                   to_string(i).c_str(), to_string(p.deleted).c_str(),
                   to_string(p.persisted).c_str(), to_string(p.copied).c_str(),
                   to_string(p.invented).c_str(), to_string(p.from_one).c_str(),
                   to_string(p.from_many).c_str(),
                   to_text("%d-%d", (int)p.most_sources, (int)p.most_landings).c_str(),
                   to_string(p.exact_covered_cells).c_str(), to_string(p.created_cells).c_str()));
    }

  };

  const int ty_top = ty;
  measuring = true;  paint_the_facts();  measuring = false;
  const int measured_h = (ty - ty_top) + pad * 3;
  ty = ty_top;
  fill_rect(context, 0, x0, y0, panel_w, measured_h);
  draw_border(context, x0, y0, panel_w, measured_h, max(1, scale));
  paint_the_facts();
}

inline void draw_op_palette(vulkan_context *context, const map<uint32_t, display_grid> &font,
                            global_app_state &app_state, const frame_layout &lay) {
  int scale = lay.scale;
  int pad = 2 * scale;
  int box_h = 5 * scale + pad * 2;
  int gap = pad * 2;
  int line_h = box_h + gap;

  // tab_names/tab_actions are indexed by op_role ORDINAL (must not move â€” hash and
  // active-tab int depend on it). Sentence display order lives in `sentence` below.
  static const char *tab_names[(int)op_role::count] = {
    "GET", "REDUCE", "SELECT", "TRANSLATE", "MAP", "GENERATE", "COMBINE"};
  static const ui_action tab_actions[(int)op_role::count] = {
    ui_action::op_tab_get, ui_action::op_tab_reduce, ui_action::op_tab_select,
    ui_action::op_tab_translate, ui_action::op_tab_map, ui_action::op_tab_generate,
    ui_action::op_tab_combine};
  int active = app_state.op_tab.load();

  // --- Role grouping (grammar: GET SELECT ACT COMBINE) ----------------------
  // A two-level read laid over the ordinal-frozen op_role enum: the sentence has
  // four parts, and ACT has kinds. Pure draw-time table, no enum touched. Each
  // part lists its member tabs in display order; one member draws no second row.
  struct sentence_part { const char *name; vector<op_role> kinds; };
  static const sentence_part sentence[] = {
    {"GET",     {op_role::get}},
    {"SELECT",  {op_role::select}},
    {"ACT",     {op_role::reduce, op_role::translate, op_role::map, op_role::generate}},
    {"COMBINE", {op_role::combine}},
  };
  constexpr int part_count = (int)(sizeof(sentence) / sizeof(sentence[0]));
  // Which role owns the active tab, so its kinds show on the sub-row.
  int active_part = 0;
  for (int r = 0; r < part_count; ++r)
    for (op_role k : sentence[r].kinds)
      if ((int)k == active) active_part = r;

  int hmargin = lay.base_margin * 3; // tripled horizontal edge padding
  auto rows_for = [&](const vector<string> &labels) {
    int rows = 1, cx = hmargin;
    for (auto &s : labels) {
      int bw = text_w(s, scale) + pad * 2;
      if (cx + bw > lay.screen_w - hmargin) { cx = hmargin; ++rows; }
      cx += bw + pad;
    }
    return rows;
  };

  // One tab's ops, alphabetical, labelled exactly the way they draw.
  int n = (int)node_type::type_count;
  auto labels_for = [&](int tab, vector<int> *out_ids = nullptr) {
    vector<int> ids;
    for (int i = 0; i < n; ++i)
      if ((int)role_of((node_type)i) == tab) ids.push_back(i);
    sort(ids.begin(), ids.end(), [](int a, int b) {
      return string(type_name[a]) < string(type_name[b]);
    });
    vector<string> labels;
    for (int i : ids) {
      string s = type_name[i];
      if (is_annotated((node_type)i))
        s += to_text("[%d-%d]", row_of((node_type)i).low, row_of((node_type)i).high);
      labels.push_back(s);
    }
    if (out_ids) *out_ids = ids;
    return labels;
  };

  vector<string> part_labels;
  for (int r = 0; r < part_count; ++r) part_labels.push_back(sentence[r].name);
  // Sub-kind row: only drawn when the active role has more than one kind.
  bool has_subrow = sentence[active_part].kinds.size() > 1;
  vector<string> sub_labels;
  if (has_subrow) for (op_role k : sentence[active_part].kinds) sub_labels.push_back(tab_names[(int)k]);
  vector<int> op_ids;
  vector<string> op_labels = labels_for(active, &op_ids);

  // The drawer is as deep as its deepest tab, and always reserves the sub-kind
  // row. Its size belongs to the ops it holds -- it used to be floored on where
  // the header ended, which made how much you could see depend on something with
  // nothing to do with ops, and a drawer that resizes as you switch tabs moves
  // the handle out from under the cursor that is holding it.
  int deepest_ops = 0;
  for (int t = 0; t < (int)op_role::count; ++t)
    deepest_ops = max(deepest_ops, rows_for(labels_for(t)));
  int deepest_sub = 0;
  for (int r = 0; r < part_count; ++r) {
    if (sentence[r].kinds.size() <= 1) continue;
    vector<string> names;
    for (op_role k : sentence[r].kinds) names.push_back(tab_names[(int)k]);
    deepest_sub = max(deepest_sub, rows_for(names));
  }
  int tab_rows = rows_for(part_labels) + deepest_sub;
  int panel_h = lay.base_margin + (tab_rows + deepest_ops) * line_h + gap;

  // Drawer model: a fixed-height panel that lives ABOVE the screen and slides
  // down. op_drawer_h = how far the bottom edge is pulled down (dragged). Panel
  // top = bottom - panel_h (negative until fully pulled).
  app_state.op_drawer_h_max.store(panel_h);
  int drawer_bottom = app_state.op_drawer_h.load();
  // Open means all the way out: a release already snaps to full or shut, so a
  // partly-open default would be a depth you can never drag back to.
  if (drawer_bottom <= 0) drawer_bottom = panel_h;
  if (drawer_bottom > panel_h) drawer_bottom = panel_h; // can't pull past full
  int panel_top = drawer_bottom - panel_h;
  app_state.op_drawer_bottom.store(drawer_bottom);

  // Opaque backdrop for the visible slice (from y=0 down to the bottom edge).
  fill_rect(context, solid_sheets.count(11) ? 11 : 0, 0, 0, lay.screen_w, drawer_bottom);
  // Borders: top at the panel top (may be above y=0), bottom at the visible
  // edge, sides spanning the visible slice.
  if (solid_sheets.count(UI_COLOR_LINE)) {
    int t = drawer_border_t(scale), off = solid_sheets[UI_COLOR_LINE].at.offset;
    if (panel_top >= 0)
      context->draw(off, lay.screen_w, t, 0, panel_top, 1, 1);        // top
    context->draw(off, lay.screen_w, t, 0, drawer_bottom - t, 1, 1);  // bottom
    context->draw(off, t, drawer_bottom, 0, 0, 1, 1);                 // left
    context->draw(off, t, drawer_bottom, lay.screen_w - t, 0, 1, 1);  // right
  }

  // Equal vertical gaps: top-border -> tabs -> separator -> ops.
  int vgap = gap;

  // Content laid out normally from the panel top (which may be above y=0, so the
  // top-most rows scroll in last as it's pulled down).
  int header_y = panel_top + drawer_border_t(scale) + vgap;

  // op_role -> its ui_action, so a sub-kind click reuses the existing handler.
  auto action_for = [&](op_role t) {
    for (int i = 0; i < (int)op_role::count; ++i)
      if ((int)t == i) return tab_actions[i];
    return ui_action::none;
  };
  // Draw one centered, wrapping row of tab boxes; returns the y of the next row.
  auto draw_tab_row = [&](const vector<op_role> &kinds, const vector<const char*> &labels,
                          int start_y, auto is_active, auto is_debt) {
    int total = 0;
    for (size_t k = 0; k < labels.size(); ++k)
      total += text_w(labels[k], scale) + pad * 2 + (k ? pad : 0);
    int x = max(hmargin, (lay.screen_w - total) / 2);
    int yy = start_y;
    for (size_t k = 0; k < labels.size(); ++k) {
      int tw = text_w(labels[k], scale) + pad * 2;
      if (x + tw > lay.screen_w - hmargin) { x = hmargin; yy += line_h; }
      if (yy >= 0) {
        int col = is_active(k) ? 1 : is_debt(k) ? 2 : 5;
        draw_boxed_text(context, font, labels[k], x, yy, scale, tw, box_h, col);
        app_state.click_zones.push_back({x, yy, tw, box_h, action_for(kinds[k])});
      }
      x += tw + pad;
    }
    return yy;
  };

  // --- Role row: GET SELECT ACT COMBINE -------------------------------------
  vector<op_role> part_first; vector<const char*> part_names;
  for (int r = 0; r < part_count; ++r) { part_first.push_back(sentence[r].kinds.front()); part_names.push_back(sentence[r].name); }
  int tab_y = draw_tab_row(part_first, part_names, header_y,
    [&](size_t r){ return (int)r == active_part; },
    [&](size_t){ return false; });  // no debt tabs left â€” nothing draws red

  // --- Sub-kind row (only when the active role has more than one kind) -------
  if (has_subrow) {
    const auto &kinds = sentence[active_part].kinds;
    vector<const char*> knames; for (op_role k : kinds) knames.push_back(tab_names[(int)k]);
    tab_y = draw_tab_row(kinds, knames, tab_y + line_h,
      [&](size_t k){ return (int)kinds[k] == active; },
      [&](size_t){ return false; });  // no debt kinds left
  }

  // --- Separator (equidistant below tabs) + ops -----------------------------
  int sep_y = tab_y + box_h + vgap;
  if (sep_y >= 0 && solid_sheets.count(5))
    context->draw(solid_sheets[5].at.offset, lay.screen_w - 2 * hmargin, 1,
                  hmargin, sep_y, 1, 1);
  int list_top = sep_y + vgap;
  int avail = lay.screen_w - 2 * hmargin;
  // Group ops into rows first, so each row can be horizontally centered.
  struct op_box { string label; int box_w; int id; };
  vector<vector<op_box>> rows(1);
  int row_w = 0;
  for (size_t i = 0; i < op_labels.size(); ++i) {
    int bw = text_w(op_labels[i], scale) + pad * 2;
    int add = bw + (rows.back().empty() ? 0 : pad);
    if (row_w + add > avail && !rows.back().empty()) { rows.emplace_back(); row_w = 0; add = bw; }
    rows.back().push_back({op_labels[i], bw, op_ids[i]});
    row_w += add;
  }
  int y = list_top;
  for (auto &row : rows) {
    int total = 0;
    for (auto &c : row) total += c.box_w;
    total += (int)(row.size() - 1) * pad;
    int x = max(hmargin, (lay.screen_w - total) / 2);
    for (auto &c : row) {
      // What inference decided, the palette says: all of an op yellow, some of
      // it orange, none of it the ordinary blue.
      inferred_amount inferred = how_much_is_inferred((node_type)c.id);
      int col = inferred == inferred_amount::all  ? (int)arc_color::yellow
              : inferred == inferred_amount::some ? (int)arc_color::orange : 1;
      if (y >= 0)
        draw_boxed_text(context, font, c.label, x, y, scale, c.box_w, box_h, col);
      x += c.box_w + pad;
    }
    y += line_h;
  }
}

// Main frame draw - orchestrates header and content area
inline void draw_frame(vulkan_context *context, const map<uint32_t, display_grid> &font,
                       const display_grid &background, const solver_state &state,
                       global_app_state &app_state, const puzzle &task,
                       int screen_w, int screen_h, float fps,
                       int puzzle_idx, int puzzle_count,
                       vector<int> &gpu_buffer) {

  frame_layout lay = compute_frame_layout(screen_w, screen_h);

  float elapsed = get_elapsed(app_state, state.search);

  app_state.click_zones.clear();
  app_state.slider_zones.clear();
  {
    lock_guard<mutex> lk(app_state.node_zones_mtx);
    app_state.node_zones.clear();

    // The header takes as many lines as its pieces need, so everything under it
    // is placed from where it actually ended, not from a reserved row count.
    lay.grid_labels_y = draw_header_rows(context, font, app_state, lay,
                                         task, puzzle_idx, puzzle_count, fps);
    int grid_label_h = 5 * lay.scale + 2 * 2 * lay.scale; // matches label box height
    lay.grid_data_start_y = lay.grid_labels_y + grid_label_h + lay.v_gap;

    // Reflow the BEST footer for a multi-line (wrapped) pipeline: a deep program
    // stacks upward (and shrinks the grid area) instead of running off the
    // right edge.
    {
      int fscale = lay.scale;
      int line_h = 9 * fscale + lay.v_gap;
      // Four sentences of room, always, so the footer does not walk up and down
      // the screen as a program grows a sentence.
      const bool whole = app_state.show_whole_programs.load();
      int best_lines = max((int)PIPELINE_FOLDED_LINES, task.best_tree ? count_pipeline_lines(task.best_tree, whole) : 1);
      lay.best_y = lay.elapsed_y - best_lines * line_h;
      // Safety: never let the footer climb into the grid-label band (absurdly
      // deep programs); the grid just shrinks to whatever room remains.
      if (lay.best_y < lay.grid_data_start_y + line_h)
        lay.best_y = lay.grid_data_start_y + line_h;
    }

    draw_footer_rows(context, font, app_state, lay, task, elapsed);
  }

  if (!draw_grid_view(context, font, background, task, state.views, lay))
    return;

  // Hover preview: draw the subtree output for every train+test example,
  // overlaid on the right half (cols 2-3). Only the solver's
  // process can run a program, so a subtree not seen yet is asked for, and
  // drawn once the answer is back.
  {
    int p_idx = state.puzzle_idx.load();
    if (state.preview_last_puzzle_idx != p_idx) {
      state.preview_cache.clear();
      state.preview_uploaded_hash = 0;
      state.preview_last_puzzle_idx = p_idx;
    }

    uint64_t h = app_state.hovered_hash.load();
    node *root = task.best_tree;
    int total_examples = (int)(task.train.size() + task.test.size());
    if (h != 0 && root && total_examples > 0) {
      auto it = state.preview_cache.find(h);
      if (it == state.preview_cache.end() && state.preview_wanted != h) {
        if (node *target = find_node_by_hash(root, h)) {
          state.preview_wanted = h;
          state.preview_program = environment::stringify_tree_via_flat(target);
        }
      }
      if (it != state.preview_cache.end() && state.preview_uploaded_hash != h) {
        const auto &payload = it->second;
        int n = min((int)payload.accumulators.size(), (int)state.preview_grids.size());
        for (int i = 0; i < n; ++i) {
          const grid &acc = payload.accumulators[i];
          if (acc.width > 0 && acc.height > 0)
            update_display_buffer(state.preview_grids[i], acc, gpu_buffer);
          else {
            state.preview_grids[i].width = 0;
            state.preview_grids[i].height = 0;
          }
          // Object-list slots for this example.
          auto &slots = state.preview_object_grids[i];
          int n_obj = min((int)payload.object_lists[i].size(), (int)slots.size());
          for (int k = 0; k < n_obj; ++k)
            update_display_buffer(slots[k], payload.object_lists[i][k], gpu_buffer);
          for (int k = n_obj; k < (int)slots.size(); ++k) {
            slots[k].width = 0;
            slots[k].height = 0;
          }
        }
        state.preview_uploaded_hash = h;
        app_state.needs_upload.store(true);
      }
      if (state.preview_uploaded_hash == h) {
        // Re-derive row Y positions exactly as draw_grid_view does, so each
        // preview overlays its own example.
        int pad = lay.base_margin / 2;
        int available_grid_h = lay.best_y - lay.grid_data_start_y;
        int slot_h = available_grid_h / total_examples;
        int col_w = lay.screen_w / 4;
        int outline = lay.scale;
        int n = min((int)state.preview_grids.size(), total_examples);

        // Backdrop spans cols 2 and 3 for every example row, giving the
        // preview ~half the screen width to lay out accumulator + object
        // strip. It also extends upward over the header row, so the preview
        // takes the whole right half.
        int back_x = 2 * col_w + lay.base_margin;
        int back_w = 2 * col_w - lay.base_margin * 2;

        // Cover headers once, above the first row, and place a single
        // "PREVIEW" label centered over the doubled column.
        {
          int hdr_y = lay.grid_labels_y;
          int hdr_h = lay.grid_data_start_y - lay.grid_labels_y;
          const display_grid &back_sheet = solid_sheets.count(0) ? solid_sheets[0] : background;
          // stride=1, not the sheet's real stride: the shader reads
          // cells[offset + row*stride + col], so a fill wider than the sheet
          // (back_w ~1256 vs the 256-wide solid sheet) walks past the block
          // into the next color sheet (blue) and leaks a band â€” visible once
          // the area is big enough (2K, not 1080). With stride=1 every cell
          // reads offset+row+col, always inside the solid block; positions are
          // computed independently so the rectangle still fills correctly.
          context->draw(back_sheet.at.offset, back_w, hdr_h,
                        back_x, hdr_y, 1, 1);

          int grid_pad = 2 * lay.scale;
          int lw = text_w("PREVIEW", lay.scale) + grid_pad * 2;
          int lh = 5 * lay.scale + grid_pad * 2;
          // Flush-left at column 2's label x.
          int label_x = 2 * col_w + lay.base_margin;
          draw_boxed_text(context, font, "PREVIEW", label_x, lay.grid_labels_y,
                          lay.scale, lw, lh, 5);
        }

        // A row with no room inside its pad shows no preview.
        for (int i = 0; i < n && slot_h > pad * 2; ++i) {
          const display_grid &m = state.preview_grids[i];
          int row_y = lay.grid_data_start_y + i * slot_h;
          int back_y = row_y + pad;
          int back_h = slot_h - pad * 2;

          // Black backdrop covers cols 2 and 3 for this row. Use color 0
          // (black in the ARC palette) rather than the cell-outline grid,
          // which is a non-black UI color.
          const display_grid &back_sheet = solid_sheets.count(0) ? solid_sheets[0] : background;
          // stride=1: solid fill wider than the 256-wide sheet would otherwise
          // overrun into the next color sheet and leak a blue band (see header
          // backdrop above).
          context->draw(back_sheet.at.offset, back_w, back_h,
                        back_x, back_y, 1, 1);

          int x_inner = back_x + outline;
          int y_top = back_y + outline;
          int inner_w = back_w - outline * 2;
          int y_used = y_top;

          // 1) Accumulator preview on top of the backdrop, with the same grey
          // outline frame the regular grid cells use.
          if (m.width > 0 && m.height > 0) {
            string dim = to_text("%02dx%02d", m.width, m.height);
            int dim_w = text_w(dim, lay.scale) + outline * 2;
            int half_h = (back_h - outline * 2) / 2;
            int cell_scale = min((inner_w - dim_w) / m.width, half_h / m.height);
            if (cell_scale < 1) cell_scale = 1;
            int gw = m.width * cell_scale, gh = m.height * cell_scale;
            context->draw(background.at.offset, gw + outline * 2, gh + outline * 2,
                          x_inner - outline, y_top - outline, 1, background.at.stride);
            context->draw(m.at.offset, m.width, m.height, x_inner, y_top, cell_scale, m.at.stride);
            draw_text(context, font, dim, x_inner + gw + outline * 2, y_top, lay.scale);
            y_used = y_top + gh + lay.v_gap * 2;
          }

          // 2) Object-list strip beneath the accumulator, one tile per object.
          // Pick a uniform scale that lets every tile fit in the strip width,
          // so adding more objects shrinks them rather than truncating.
          const auto &slots = state.preview_object_grids[i];
          int strip_h = (back_y + back_h) - y_used - outline;
          if (strip_h > 4) {
            int max_obj_h = 0;
            int sum_obj_w = 0;
            int n_obj = 0;
            for (const display_grid &og : slots) {
              if (og.width <= 0 || og.height <= 0) continue;
              max_obj_h = max(max_obj_h, (int)og.height);
              sum_obj_w += og.width;
              ++n_obj;
            }
            if (n_obj > 0 && max_obj_h > 0) {
              int gap = lay.v_gap * 2;
              int total_gap = gap * (n_obj - 1);
              int avail_w = inner_w - total_gap;
              int s_w = avail_w / max(1, sum_obj_w);
              int s_h = strip_h / max_obj_h;
              int cell_scale = max(1, min(s_w, s_h));
              int x_cur = x_inner;
              const display_grid &line_sheet = solid_sheets.count(UI_COLOR_LINE)
                                          ? solid_sheets[UI_COLOR_LINE]
                                          : background;
              int line_th = max(1, lay.scale / 2);
              for (size_t bi = 0; bi < slots.size(); ++bi) {
                const display_grid &og = slots[bi];
                if (og.width <= 0 || og.height <= 0) continue;
                int gw = og.width * cell_scale, gh = og.height * cell_scale;
                context->draw(background.at.offset, gw + outline * 2, gh + outline * 2,
                              x_cur - outline, y_used - outline, 1, background.at.stride);
                context->draw(og.at.offset, og.width, og.height, x_cur, y_used, cell_scale, og.at.stride);
                auto cache_it = state.preview_cache.find(h);
                if (cache_it != state.preview_cache.end()
                    && bi < cache_it->second.object_bboxes[i].size()) {
                  const auto &bb = cache_it->second.object_bboxes[i][bi];
                  int bx = x_cur + bb.x0 * cell_scale;
                  int by = y_used + bb.y0 * cell_scale;
                  int bw = (bb.x1 - bb.x0 + 1) * cell_scale;
                  int bh = (bb.y1 - bb.y0 + 1) * cell_scale;
                  int t = min(line_th, min(bw, bh));
                  if (t > 0) {
                    context->draw(line_sheet.at.offset, bw, t,
                                  bx, by, 1, line_sheet.at.stride);
                    context->draw(line_sheet.at.offset, bw, t,
                                  bx, by + bh - t, 1, line_sheet.at.stride);
                    context->draw(line_sheet.at.offset, t, bh,
                                  bx, by, 1, line_sheet.at.stride);
                    context->draw(line_sheet.at.offset, t, bh,
                                  bx + bw - t, by, 1, line_sheet.at.stride);
                  }
                }
                x_cur += gw + gap;
              }
            }
          }
        }
      }
    }
  }

  // What inference read off the puzzle. Coarseness and line both wrap against
  // what this puzzle actually offers, which only the readout knows, and they
  // step both ways, so the wrap must carry negatives round to the top. Counted
  // once here, for the wrap and for the pane.
  if (int view = app_state.inference_view.load(); view != global_app_state::INFERENCE_OFF) {
    const auto &correspondence = task.inf.correspondence;
    // THE STEPPER WALKS THE CUTS, each at the one reading it kept: the
    // combination of freedoms that entered the election for it. Told which cut
    // inference chose and how many there are, it hands back the one to show —
    // following the chosen one until somebody steps away.
    vector<int> kept;
    for (size_t method = 0; method < correspondence.best_of.size(); ++method)
      if (correspondence.best_of[method] >= 0 &&
          correspondence_election::competes((segmentation_methods)method))
        kept.push_back(correspondence.best_of[method]);
    const int elected_cut = (int)(ranges::find(kept, (int)correspondence.chosen) - kept.begin());
    const int cut = app_state.inference_rank.pick(elected_cut, max(1, (int)kept.size()));
    const int rank = kept.empty() ? 0 : kept[min(cut, (int)kept.size() - 1)];
    // WALKED FOR THE ONE READING ON SCREEN, and worked out here rather than
    // stored: these are places in the facts, not copies of them.
    static const vector<correspondence_election::pair_info> nothing;
    const auto &pairs = rank < (int)correspondence.readings.size()
                          ? correspondence.readings[rank].pairs : nothing;
    reading_on_screen walked;
    walked.pairs = &pairs;
    walked.pieces = pieces_of(pairs);
    walked.alone = pieces_standing_alone(pairs, walked.pieces);
    walked.refused = refusals_of(pairs);
    walked.left_over = left_over_of(pairs);
    walked.lines = lines_of(pairs);
    walked.sets = motif_sets(pairs, walked.pieces);
    const auto& pieces = walked.pieces;
    const auto& alone = walked.alone;
    const auto& refused = walked.refused;
    const auto& left_over = walked.left_over;
    const int drawn = (int)walked.lines.size();
    // Each track is told how long its list is and hands back what is chosen.
    const int pieces_n = (int)alone.size();
    const int dropped_n = (int)refused.size();
    const int step = app_state.inference_edge.pick(drawn);
    const int piece_at = app_state.inference_piece.pick(pieces_n);
    const int dropped_at = app_state.inference_dropped.pick(dropped_n);
    const auto& sets = walked.sets;
    const int motif_at = app_state.inference_motif.pick((int)sets.size());
    const int left_over_n = (int)left_over.size();
    const int left_over_at = app_state.inference_left_over.pick(left_over_n);
    if (view == global_app_state::INFERENCE_LINES) {
      // DRAWN IN THE ORDER THE SLIDERS ARE LISTED, because that is the order
      // the outlines nest in when several land on one object.
      nested_outlines rings;
      if (step != SHOW_OFF)
        draw_correspondence(context, task, lay, rank, step, rings);
      // The refused match where it wanted to go, in red.
      for (int i = 0; i < dropped_n; ++i) {
        if (dropped_at != SHOW_ALL && dropped_at != i) continue;
        const row_place &at = refused[i];
        const auto &dr = pairs[at.pair].discarded[at.which].match;
        const line_ends ends = ends_of(pairs[at.pair], dr);
        const bool one = (dropped_at == i);
        // ONE shows the refused piece and where it wanted to go; ALL shows only
        // the wanted landings. So the first box is on the piece's own grid when
        // one is singled out and on the other grid otherwise.
        draw_selection(context, task, lay, at.pair, one == dr.from_input,
                       one ? ends.from_x : ends.to_x, one ? ends.from_y : ends.to_y,
                       one ? ends.from_w : ends.to_w, one ? ends.from_h : ends.to_h,
                       one, ends.to_x, ends.to_y, ends.to_w, ends.to_h,
                       (int)arc_color::red, rings);
      }
      // COPIES OF ONE THING, ONE SET AT A TIME. A grid can hold several
      // motifs and one color cannot say which box belongs with which, so the
      // set is stepped like everything else. MAGENTA IS THE MEDOID and blue the
      // rest: a set wants an exemplar, and it cannot be the biggest or the
      // first — both pick by luck, and a block the noise landed on is complete
      // and still the odd one out.
      for (const piece_place &at : pieces) {
        const auto &pc = piece_in(pairs, at);
        if (pc.motif < 0) continue;
        if (motif_at != SHOW_ALL &&
            !(motif_at < (int)sets.size() && piece_is_in(pc, at, sets[motif_at]))) continue;
        draw_selection(context, task, lay, at.pair, at.from_input,
                       pc.summary.col, pc.summary.row, pc.summary.width, pc.summary.height, false, 0, 0, 0, 0,
                       (int)(pc.medoid ? arc_color::magenta : arc_color::blue),
                       rings);
      }
      // A BOX AND NO LINE. These are the pieces nothing partners, so there is no
      // other end to join to — but there is a place: the patch of nothing they
      // cover on the other grid, drawn there so the two are seen together. That
      // patch is where a grid makes something out of nothing.
      for (int i = 0; i < pieces_n; ++i) {
        if (piece_at != SHOW_ALL && piece_at != i) continue;
        const piece_place &at = alone[i];
        const auto &pc = piece_in(pairs, at);
        draw_selection(context, task, lay, at.pair, at.from_input,
                       pc.summary.col, pc.summary.row, pc.summary.width, pc.summary.height, pc.blank.found(),
                       pc.blank.box.col, pc.blank.box.row, pc.blank.box.width, pc.blank.box.height,
                       (int)arc_color::gray, rings);
      }
      // THE CELLS NOTHING ACCOUNTED FOR, one cell each, so there is never room
      // to step inside another outline — they are drawn last and simply sit
      // where they are.
      for (int i = 0; i < left_over_n; ++i) {
        if (left_over_at != SHOW_ALL && left_over_at != i) continue;
        const row_place &at = left_over[i];
        const auto &cell = pairs[at.pair].left_over[at.which];
        draw_selection(context, task, lay, at.pair, cell.from_input,
                       cell.col, cell.row, 1, 1, false, 0, 0, 0, 0,
                       (int)arc_color::orange, rings);
      }
    }
    draw_inference_pane(context, font, app_state, task, lay, walked, view, rank, cut,
                        (int)kept.size(), step, drawn, piece_at, dropped_at, motif_at, left_over_at);
  }

  // The op palette goes last, so it is over everything. It already swallows every
  // click that lands under it, so drawing it under anything made the grid
  // disagree with what the mouse was doing. The pull-handle is always drawn
  // (discoverable when closed, draggable open), riding the drawer's bottom edge:
  // at y=0 when closed, at the drawer bottom when open.
  bool op_open = app_state.show_op_list.load();
  if (op_open)
    draw_op_palette(context, font, app_state, lay);
  draw_op_handle(context, app_state, lay, op_open ? app_state.op_drawer_bottom.load() : 0);
}

inline void load_font_atlas(vector<int> &gpu_buffer, map<uint32_t, display_grid> &chars) {
  string charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZx.eE+ -() :<>,_/[ ]|";
  vector<uint32_t> cps;
  cps.reserve(128);
  for (char ch : charset) cps.push_back((unsigned char)ch);
  cps.push_back(0x25B2); // triangle up
  cps.push_back(0x25BC); // triangle down
  cps.push_back(0xE000); // speaker on  (mute toggle)
  cps.push_back(0xE001); // speaker muted
  cps.push_back(0xE002); // copy icon
  cps.push_back(0xE003); // paste icon

  for (uint32_t cp : cps) {
    pad_buffer(gpu_buffer);
    display_grid g = create_character_grid(cp);
    g.at.offset = (int)gpu_buffer.size();
    g.at.stride = 4;
    for (int r = 0; r < 5; ++r) {
      for (int c = 0; c < 3; ++c)
        gpu_buffer.push_back(g.data[r][c]);
      gpu_buffer.push_back(0);
    }
    chars[cp] = g;
  }
  vector<int> colors;
  colors.reserve(64);
  for (int i = 0; i < COLOR_COUNT; ++i)
    colors.push_back(i);
  for (int i = 0; i < COLOR_COUNT; ++i)
    colors.push_back(UI_COLOR_PASTEL_START + i);
  // Add Line Color
  colors.push_back(UI_COLOR_LINE);

  for (int c_idx : colors) {
    pad_buffer(gpu_buffer);
    solid_sheets[c_idx] = display_grid{{256, 256}};
    solid_sheets[c_idx].at.offset = (int)gpu_buffer.size();
    solid_sheets[c_idx].at.stride = 256;
    gpu_buffer.insert(gpu_buffer.end(), 256 * 256, c_idx);
    for (auto const &[k, v] : chars) {
      if ((k < 32 || k > 126) && k != 0x25B2 && k != 0x25BC &&
          k != 0xE000 && k != 0xE001 && k != 0xE002 && k != 0xE003)
        continue;
      pad_buffer(gpu_buffer);
      display_grid cv = v;
      cv.at.offset = (int)gpu_buffer.size();
      cv.at.stride = 4;
      for (int r = 0; r < cv.height; ++r) {
        for (int c = 0; c < cv.width; ++c) {
          if (c_idx < UI_COLOR_PASTEL_START) {
            // Label Mode: Colored letters on transparent background
            gpu_buffer.push_back(v.data[r][c] == UI_COLOR_LINE ? c_idx : -1);
          } else {
            // Box Mode: White letters on colored background
            gpu_buffer.push_back(v.data[r][c] == UI_COLOR_LINE ? UI_COLOR_LINE : c_idx);
          }
        }
        gpu_buffer.push_back(0);
      }
      colored_fonts[c_idx][k] = cv;
    }
  }
}

}
