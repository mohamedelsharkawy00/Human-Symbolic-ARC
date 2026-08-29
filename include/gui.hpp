#pragma once
#include "environment.hpp"
#include "grid_buffer.hpp"
#include "global_app_state.hpp"
#include "cuda_dsl.cuh"
#include "vulkan_context.hpp"
#include "sound_manager.hpp"

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

constexpr int UI_COLOR_LINE = (int)arc_color::count;
constexpr int UI_COLOR_PASTEL_START = (int)arc_color::count * 2;

inline grid create_character_grid(uint32_t ch) {
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
  grid g;
  g.height = 5;
  g.width = 3;
  g.stride = 3;
  if (glyphs.count(ch)) {
    int v = glyphs[ch];
    for (int row_idx = 0; row_idx < 5; ++row_idx) {
      int row_bits = (v >> (3 * (4 - row_idx))) & 7;
      for (int col_idx = 0; col_idx < 3; ++col_idx) {
        g.data[row_idx][col_idx] = (row_bits & (1 << (2 - col_idx))) ? 10 : 0;
      }
    }
  }
  return g;
}

// Global cache for colored fonts: [ColorIdx][Codepoint] -> Grid
inline map<int, map<uint32_t, grid>> colored_fonts;
// Global cache for solid color sheets: [ColorIdx] -> Grid
inline map<int, grid> solid_sheets;

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

inline void draw_text(vulkan_context *context, const map<uint32_t, grid> &font,
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
      const grid &g = font.at(cp);
      context->draw(g.offset, g.width, g.height, cur_x, y, scale, g.stride);
      cur_x += (g.width + 1) * scale;
    } else if (cp == ' ') {
      cur_x += 4 * scale;
    } else {
      uint32_t up = toupper((unsigned char)cp);
      if (font.count(up)) {
        const grid &g = font.at(up);
        context->draw(g.offset, g.width, g.height, cur_x, y, scale, g.stride);
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
  int off = solid_sheets[color].offset;
  const int tile = 200;   // comfortably inside the sheet
  for (int fx = 0; fx < w; fx += tile)
    context->draw(off, min(tile, w - fx), h, x + fx, y, 1, 1);
}

// Draws a centered text label inside a colored background box.
inline void draw_boxed_text(vulkan_context *context,
                            const map<uint32_t, grid> &font, const string &text,
                            int x, int y, int scale, int box_w, int box_h,
                            int color_idx) {
  // Retrieve Pre-allocated Solid Sheets
  int pastel_idx = UI_COLOR_PASTEL_START + color_idx;
  const grid *bg_grid = solid_sheets.count(pastel_idx)
                            ? &solid_sheets[pastel_idx]
                            : &solid_sheets[color_idx];

  context->draw(bg_grid->offset, box_w, box_h, x, y, 1, bg_grid->stride);

  int text_x = x + (box_w - text_w(text, scale)) / 2;
  int text_y = y + (box_h - 5 * scale) / 2;

  const map<uint32_t, grid> &box_font =
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

inline string format_sci(float val) {
  if (!isfinite(val)) return string("-------");
  float a_val = abs(val);
  if (a_val < 1e-9f) return string("0000.00 e0");

  int exp = (int)floor(log10(a_val));
  int eng_exp = (exp >= 0) ? (exp / 3) * 3 : ((exp - 2) / 3) * 3;
  float eng_base = val / pow(10.0f, (float)eng_exp);

  return fmt::format("{:07.2f} e{}", eng_base, eng_exp);
}

struct frame_layout {
  int base_margin, scale, v_gap;
  int row1_y;
  // Set once the header has drawn and knows how many lines it needed.
  int grid_labels_y, grid_data_start_y;
  int footer1_y, footer2_y, footer3_y;
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

  // Footer: CURRENT, BEST, ELAPSED top-down
  l.footer3_y = screen_h - l.base_margin - 9 * l.scale;
  l.footer2_y = l.footer3_y - row_h;
  l.footer1_y = l.footer2_y - row_h;

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

// Number of wrapped lines a pipeline needs for a given horizontal budget. MUST
// mirror draw_pipeline's token advance exactly so reflow and draw agree.
inline int count_pipeline_lines(node *root, int start_x, int max_x, int scale, int pad) {
  vector<node *> ops;
  collect_pipeline_ops(root, ops);
  int gap = pad, x = start_x, lines = 1;
  for (size_t i = 0; i < ops.size(); ++i) {
    int bw = text_w(get_node_label(ops[i]->type, ops[i]->value), scale) + pad * 2;
    if (i > 0 && x + bw > max_x) { x = start_x; ++lines; }
    x += bw + gap;
  }
  return lines;
}

// Draws the header: every control and stat, in reading order. Returns the
// bottom of the last line it needed, so what sits under it knows where to start.
inline int draw_header_rows(
    vulkan_context *context,
    const map<uint32_t, grid> &font,
    global_app_state &app_state,
    const frame_layout &lay,
    const puzzle &task,
    int puzzle_idx, int puzzle_count,
    long long current_attempts, float aps, float fps, float gpu_ms,
    int cur_depth, int best_depth, int search_depth_limit) {

  int scale = lay.scale;

  // Format display strings.
  // Width the index/count to the digit-count of puzzle_count so the value never
  // outgrows its reserved slot (e.g. 1000/1000 on the training set, which used
  // to overflow the fixed 3-digit reservation and shove the >/>| buttons onto
  // the number).
  int puz_digits = (int)to_string(puzzle_count > 0 ? puzzle_count : 1).size();
  string puz_str = fmt::format("{:0{}d}/{:0{}d}", puzzle_idx + 1, puz_digits, puzzle_count, puz_digits);
  string attempts_str = format_sci((float)current_attempts);
  string aps_str = format_sci(aps);
  string gpu_str = fmt::format("{:.0f}ms", gpu_ms);
  string fps_str = fmt::format("{:06.1f}", fps);
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
    button_piece("INFERENCE", fmt::format("{:<5} [I]", view_names[view]),
                 ui_action::cycle_inference, view ? 4 : 1);
  }

  button_piece("RESET [R]", "", ui_action::reset_solver, 1);
  button_piece("LOGGING", fmt::format("{:<3} [L]", log_state_str), ui_action::toggle_log, 1);
  button_piece(fmt::format("{:<7} [SPACE]", status_str), "", ui_action::toggle_pause, 1);
  button_piece("IMPORT [N]", "", ui_action::import_sol, 1);
  button_piece("EXPORT [E]", "", ui_action::export_sol, 1);
  button_piece(fmt::format("BENCHMARK {:<2} [B]", app_state.is_benchmarking ? "ON" : ""), "",
               ui_action::toggle_benchmark, app_state.is_benchmarking ? 4 : 1);

  {
    int ts = (int)app_state.benchmark_timeout_seconds;
    stepper_piece("TIMEOUT", fmt::format("{:02d}:{:02d}:{:02d}", ts / 3600, (ts % 3600) / 60, ts % 60),
                  text_w("88:88:88", scale), ui_action::dec_timeout, ui_action::inc_timeout);
  }
  stepper_piece("LIMIT", fmt::format("{:03d}", app_state.benchmark_puzzle_limit),
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

  // Dataset checkmarks: TRN / EVL / TST. Each toggles whether that set's tasks
  // are concatenated into the loaded list; clicking reloads the merged list live
  // (handled by the worker). On = lit box with a '+' prefix, off = dim with '-'.
  // (The font has '+'/'-' but no tick glyph, so the +/- reads the state.)
  {
    int ds = app_state.dataset_flags.load();
    int w = box_w("+TRN");
    int x = spot(w * 3 + pad * 2), y = fy;
    auto draw_ds = [&](const char *name, int flag, ui_action act) {
      bool on = (ds & flag) != 0;
      box_at(string(on ? "+" : "-") + name, x, y, w, on ? 4 : 1, act);
      x += w + pad;
    };
    draw_ds("TRN", 1 /*DS_TRAIN*/, ui_action::toggle_ds_train);
    draw_ds("EVL", 2 /*DS_EVAL*/,  ui_action::toggle_ds_eval);
    draw_ds("TST", 4 /*DS_TEST*/,  ui_action::toggle_ds_test);
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

  pair_piece("FPS", fmt::format("{:<7}", fps_str));
  pair_piece("ATTEMPTS", fmt::format("{:<12}", attempts_str));
  pair_piece("APS", fmt::format("{:<12}", aps_str));
  // How long one GPU batch takes. It sits beside APS on purpose: a frame spent
  // inside a running kernel scores no attempts, so APS alone cannot say busy
  // from stalled, and this is what tells them apart.
  pair_piece("GPU BATCH", fmt::format("{:<8}", gpu_str));
  pair_piece("BEST DEPTH", fmt::format("{:<3}", best_depth));
  pair_piece("CURRENT DEPTH", fmt::format("{:<3}", cur_depth));
  pair_piece("MAX DEPTH", fmt::format("{:<3}", search_depth_limit));

  // ARC palette reference -- a flush strip of square swatches, each with its
  // index drawn with a black outline.
  {
    int n_colors = (int)arc_color::count;
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
        const grid &sw = solid_sheets[i];
        context->draw(sw.offset, sw_side, sw_side, cx, cy, 1, sw.stride);
      }
      // Index digit: white with a black outline, centered on the swatch.
      string num = fmt::format("{}", i);
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

// Draws the bottom of the screen: the CURRENT and BEST program pipelines and the
// elapsed timer.
inline void draw_footer_rows(
    vulkan_context *context,
    const map<uint32_t, grid> &font,
    global_app_state &app_state,
    const frame_layout &lay,
    const puzzle &task,
    const solver_state &state,
    float elapsed) {

  int scale = lay.scale;
  int pad = 2 * scale;

  // Walk a unary tree from the root op outward (Input first), pushing each
  // op as its own colored token. Returns end x.
  // `top_y` is the first line; long pipelines wrap downward at the right edge
  // (continuation lines indent under the first token). Footer Y-positions are
  // reflowed in draw_frame so the wrapped block stacks without overlapping.
  auto draw_pipeline = [&](node *root, int start_x, int top_y, int source) {
    int box_h = 5 * scale + pad * 2;
    int gap = pad;
    int max_x = lay.screen_w - lay.base_margin;
    int line_h = 9 * scale + lay.v_gap; // matches frame_layout row_h
    vector<node *> ops;
    collect_pipeline_ops(root, ops);

    uint64_t hh = app_state.hovered_hash.load();
    int hs = app_state.hovered_source.load();
    int x = start_x, y = top_y;
    for (size_t i = 0; i < ops.size(); ++i) {
      node *n = ops[i];
      string label = get_node_label(n->type, n->value);
      int bw = text_w(label, scale) + pad * 2;
      if (i > 0 && x + bw > max_x) {
        x = start_x; y += line_h;
      }
      int color_idx = 1;
      if (n->hash != 0 && hh == n->hash && hs == source)
        color_idx = 4;
      draw_boxed_text(context, font, label, x, y, scale, bw, box_h, color_idx);
      app_state.node_zones.push_back({x, y, bw, box_h, n->hash, source});
      x += bw + gap;
    }
    return x;
  };

  // Footer 1: CURRENT program pipeline
  {
    int lw = text_w("CURRENT", scale) + pad * 2;
    int box_h = 5 * scale + pad * 2;
    draw_boxed_text(context, font, "CURRENT", lay.base_margin, lay.footer1_y, scale, lw, box_h, 5);
    if (state.current_tree)
      draw_pipeline(state.current_tree, lay.base_margin + lw + pad, lay.footer1_y, 2);
  }

  // Footer 2: BEST program pipeline
  {
    int lw = text_w("BEST", scale) + pad * 2;
    int box_h = 5 * scale + pad * 2;
    draw_boxed_text(context, font, "BEST", lay.base_margin, lay.footer2_y, scale, lw, box_h, 5);
    if (task.best_tree)
      draw_pipeline(task.best_tree, lay.base_margin + lw + pad, lay.footer2_y, 1);
  }

  // Footer 3: Elapsed timer
  {
    string elapsed_val = format_elapsed(elapsed);
    int lw = text_w("ELAPSED", scale) + pad * 2;
    draw_boxed_text(context, font, "ELAPSED", lay.base_margin, lay.footer3_y, scale, lw, 5 * scale + pad * 2, 5);
    draw_text(context, font, elapsed_val, lay.base_margin + lw + pad, lay.footer3_y + pad, scale);
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

inline int count_subtree_nodes(node *n) {
  int total = 0;
  for (; n; n = n->arg) ++total;
  return total;
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
  // The dimension label (and the score, on the solver columns) sits to the
  // right of the grid, so it eats into the width the grid may scale into.
  int dim_w = text_w("00x00", lay.scale) + outline * 2;
  int score_space = (col_idx >= 2) ? text_w("0000.00", lay.scale) + outline * 2 : 0;
  int side_space = max(dim_w, score_space);
  p.scale = min((col_w - pad * 2 - side_space - outline * 2) / m.width,
                (slot_h - pad * 2 - outline * 2) / m.height);
  if (p.scale < 1) return p;
  p.x = col_idx * col_w + lay.base_margin + outline;
  p.y = row_y + pad + outline;
  p.fits = true;
  return p;
}

// Draws the 4-column grid view (INPUT / OUTPUT / CURRENT / BEST).
// Returns false if there is no data to draw.
inline bool draw_grid_view(
    vulkan_context *context,
    const map<uint32_t, grid> &font,
    const grid &background,
    const puzzle &task,
    const frame_layout &lay) {

  if (task.train.empty() && task.test.empty())
    return false;

  int inputs_n = (int)(task.train.size() + task.test.size());
  int pad = lay.base_margin / 2;
  int available_grid_h = lay.footer1_y - lay.grid_data_start_y;

  int n = inputs_n > 0 ? inputs_n : 1;
  int slot_h = available_grid_h / n;
  int col_w = lay.screen_w / 4;
  int cur_y = lay.grid_data_start_y;

  auto draw_cell = [&](const grid &m, int col_idx, int row_y,
                       float score_val = SCORE_MIN - 1.0f) {
    if (m.width <= 0 || m.height <= 0 || !m.visible)
      return;

    bool is_scored_column = (col_idx >= 2);
    bool has_score = (score_val > SCORE_MIN + 0.1f);

    if (is_scored_column && !has_score)
      return;

    int outline = lay.scale;
    int info_h = 5 * lay.scale;
    string dim = fmt::format("{:02d}x{:02d}", m.width, m.height);
    grid_place p = place_grid_cell(m, col_idx, row_y, lay, col_w, slot_h);
    if (!p.fits) return; // grid band too small (e.g. footers ate the space) — skip
    int scale = p.scale;
    int grid_w = m.width * scale, grid_h = m.height * scale;
    int x = p.x, y = p.y;

    context->draw(background.offset, grid_w + outline * 2, grid_h + outline * 2,
                  x - outline, y - outline, 1, background.stride);
    context->draw(m.offset, m.width, m.height, x, y, scale, m.stride);

    int info_x = x + grid_w + outline * 2;
    draw_text(context, font, dim, info_x, y, lay.scale);

    if (has_score) {
      string s = fmt::format("{:07.2f}", abs(score_val));
      draw_text(context, font, s, info_x, y + grid_h - info_h, lay.scale);
    }
  };

  auto draw_section = [&](const vector<example> &vec) {
    if (vec.empty())
      return;
    for (const auto &ex : vec) {
      draw_cell(ex.input, 0, cur_y);
      draw_cell(ex.output, 1, cur_y);
      draw_cell(ex.current, 2, cur_y, ex.cur_score);
      draw_cell(ex.best, 3, cur_y, ex.best_score);
      cur_y += slot_h;
    }
  };

  string heads[] = {"INPUT", "OUTPUT", "CURRENT", "BEST"};
  int grid_pad = 2 * lay.scale;
  for (int i = 0; i < 4; ++i) {
    int header_label_x = i * col_w + lay.base_margin;
    int lw = text_w(heads[i], lay.scale) + grid_pad * 2;
    int lh = 5 * lay.scale + grid_pad * 2;
    draw_boxed_text(context, font, heads[i], header_label_x, lay.grid_labels_y, lay.scale, lw, lh, 5);
  }

  draw_section(task.train);
  draw_section(task.test);
  return true;
}

// Op palette overlay (toggle 'O'): a flow-laid list of every DSL op, each shown
// as Name or Name[0-N] for annotated ops. Read-only for now â€” the first piece of
// the in-GUI composer (next step: click a token to append it to a draft program
// and let the existing hover-preview engine show the effect live).
// Which tab an op is drawn under is its role, read straight off the op list
// (role_of, cuda_dsl.cuh). See notes/2 - DSL/Op Algebra.md.

// Top-edge drawer overlay: a full-width panel that floats on top of the live
// frame, covering the header band from the top of the screen down to the
// grid-label row (no shrink â€” the grids and the BEST/CURRENT/ELAPSED footers
// stay visible below). A tab row picks the op family; the active family's ops
// flow below in as many columns as fit. Opens by dragging its handle, no key;
// tabs are click_zones.
// Thickness of the drawer/handle border (quad-weight).
inline int drawer_border_t(int scale) { return max(1, scale * 2); }

// Draw a white border rectangle around a region using the line sheet. stride=1
// so wide/tall edges can't overrun the solid sheet.
inline void draw_border(vulkan_context *context, int x, int y, int w, int h, int t,
                        int color = UI_COLOR_LINE) {
  if (!solid_sheets.count(color)) return;
  int off = solid_sheets[color].offset;
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
    int off = solid_sheets[hcol].offset;
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
  int off = solid_sheets[color].offset;
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
inline vector<motif_set> motif_sets(const puzzle::correspondence_readout::reading& R) {
  vector<motif_set> sets;
  for (const auto& pc : R.pieces_seen) {
    if (pc.motif < 0) continue;
    const motif_set here{ pc.pair, pc.from_input, pc.motif };
    if (ranges::find(sets, here) == sets.end()) sets.push_back(here);
  }
  return sets;
}
inline bool piece_is_in(const puzzle::correspondence_readout::reading::piece_row& pc,
                        const motif_set& set) {
  return pc.motif >= 0 && pc.motif == set.which &&
         pc.pair == set.pair && pc.from_input == set.from_input;
}

// THE PIECES THAT STAND ALONE — the ones no piece on the other grid is. Not
// the ones nothing can be said about: deleted, inserted and covered are all
// answers, and each one is said under the stepper. What these lack is a
// partner, which is why the walk is named for that and not for silence.
inline vector<int> pieces_standing_alone(const puzzle::correspondence_readout::reading& R) {
  vector<int> which;
  for (int i = 0; i < (int)R.pieces_seen.size(); ++i)
    if (!R.pieces_seen[i].has_partner) which.push_back(i);
  return which;
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
  if (inputs_n <= 0 || rank >= (int)task.correspondence.readings.size()) return;
  int slot_h = (lay.footer1_y - lay.grid_data_start_y) / inputs_n;
  int col_w = lay.screen_w / 4;
  int t = max(1, lay.scale);
  int nth = -1;
  for (const auto &ln : task.correspondence.lines) {
    if (ln.rank != rank) continue;
    if (pick >= 0 && ++nth != pick) continue;
    if (ln.pair < 0 || ln.pair >= (int)task.train.size()) continue;
    const example &ex = task.train[ln.pair];
    int row_y = lay.grid_data_start_y + ln.pair * slot_h;
    // Input is the left column, output the right; the line says which end it
    // starts on, so the two readings draw on the same grid without arguing.
    grid_place from = place_grid_cell(ln.from_input ? ex.input : ex.output,
                                      ln.from_input ? 0 : 1, row_y, lay, col_w, slot_h);
    grid_place to   = place_grid_cell(ln.from_input ? ex.output : ex.input,
                                      ln.from_input ? 1 : 0, row_y, lay, col_w, slot_h);
    if (!from.fits || !to.fits) continue;
    draw_line(context, (int)arc_color::yellow,
              from.x + ln.from_col * from.scale / 2,
              from.y + ln.from_row * from.scale / 2,
              to.x + ln.to_col * to.scale / 2,
              to.y + ln.to_row * to.scale / 2, t);
    // The two boxes the line joins: where the piece sits, and where it landed.
    rings.add(context, from.x + ln.from_x * from.scale, from.y + ln.from_y * from.scale,
              ln.from_w * from.scale, ln.from_h * from.scale);
    // Hanging off the edge, the landing is the piece's own size, not the
    // clipped one — the window is whole even where the grid stops, and it is
    // drawn past the wall because that is what it says.
    const bool clipped = (ln.outside > 0);
    rings.add(context, to.x + ln.to_x * to.scale, to.y + ln.to_y * to.scale,
              (clipped ? ln.from_w : ln.to_w) * to.scale,
              (clipped ? ln.from_h : ln.to_h) * to.scale);
    // WHICH BIT OF IT WAS ALREADY THERE. A sits-on line says this piece is
    // under that one, and the box round the covering piece alone does not say
    // where underneath. So the piece's own outline is drawn a second time, on
    // the far grid, at its own place — a seed shows up inside the tile stamped
    // over it. Only for sits-on: every other line MOVED its piece, and drawing
    // it at its old place on the new grid would say it stayed.
    if (ln.kind == puzzle::correspondence_readout::line::sits_on)
      rings.add(context, to.x + ln.from_x * to.scale, to.y + ln.from_y * to.scale,
                ln.from_w * to.scale, ln.from_h * to.scale, UI_COLOR_LINE, true);
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
  int slot_h = (lay.footer1_y - lay.grid_data_start_y) / inputs_n;
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
// names a choice is the thing you click to change it. Parked over the
// CURRENT/BEST columns, which are empty until solving.
inline void draw_inference_pane(vulkan_context *context, const map<uint32_t, grid> &font,
                                global_app_state &app_state,
                                const puzzle &task, const frame_layout &lay,
                                int view, int rank, int step, int drawn,
                                int piece_at, int dropped_at, int motif_at,
                                int left_over_at) {
  const bool facts_face = (view == global_app_state::INFERENCE_FACTS);
  int scale = lay.scale;
  int pad = 2 * scale;
  int line_h = 7 * scale;
  int sw = 5 * scale;                 // color swatch side
  // Park it in the CURRENT/BEST region (cols 2-3), which is empty until solving.
  int col_w = lay.screen_w / 4;
  int x0 = 2 * col_w + lay.base_margin, y0 = lay.grid_data_start_y;
  int panel_w = 2 * col_w - lay.base_margin * 2;

  int t = max(1, scale / 2);          // swatch outline thickness
  auto swatch = [&](int color, int x, int y) {
    // white outline first, swatch inset on top â€” so black reads on black.
    draw_border(context, x, y, sw, sw, t);
    if (color >= 0 && color < (int)arc_color::count && solid_sheets.count(color))
      context->draw(solid_sheets[color].offset, sw - t * 2, sw - t * 2,
                    x + t, y + t, 1, solid_sheets[color].stride);
    else
      draw_text(context, font, "-", x + t, y + t, scale);
  };

  const int readings = (int)task.correspondence.readings.size();

  // Count total lines to size the backing panel.
  auto& bg = task.bg;
  auto& F = task.facts;
  // each election: name row + column-header row + candidate rows + half-gap
  auto dir_rows = [](const vector<puzzle::background_readout::cand>& d) { return (int)d.size() + 2; };
  // size + colors(4) + blank + bg head/veto/per-pair + blank + elections
  // + blank + changed(2) + blank + correspondence's two head rows + one per pair
  int total = 13 + dir_rows(bg.across_inputs) + dir_rows(bg.across_outputs) +
              dir_rows(bg.across_inputs_outputs) + (int)F.pairs.size() + 2;
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
  int panel_h = facts_face
      ? total * line_h + line_h * 2 + pad * 3   // + gaps
      // WRITTEN THE WAY THE BODY BELOW SPENDS IT, term for term, so a row added
      // there is a term added here and the frame cannot end up shorter than
      // what it holds. Six steppers, a detail line under each and a second one
      // under ALONE, two rules, the matrix (its axis title, three freedom
      // rows, a row per cut TWICE with a title over the second block, and the
      // legend), then the pair table (its head, a row per pair) and the three
      // reasons a match was refused.
      : frame_pad * 2 +
            6 * (box_h + pad) + 7 * line_h +
            2 * (block_gap * 2 + row_h - 5 * scale) +
            (6 + 2 * (int)split_kind::split_count) * row_h +
            (1 + (int)F.pairs.size()) * row_h + 3 * line_h;

  // Dark backing, so the grids behind don't read through the readout.
  fill_rect(context, 0, x0, y0, panel_w, panel_h);
  draw_border(context, x0, y0, panel_w, panel_h, max(1, scale));

  int tx = x0 + margin, ty = y0 + (facts_face ? pad : frame_pad);
  const int right_x = x0 + panel_w - margin;
  auto line = [&](const string& s) { draw_text(context, font, s, tx, ty, scale); ty += line_h; };
  // Colors as a list of digits, e.g. "0,2,4"; a dash when the set is empty.
  auto color_list = [](uint16_t m) {
    string s;
    for (int i = 0; i < 10; ++i)
      if (m & (1u << i)) { if (!s.empty()) s += ','; s += (char)('0' + i); }
    return s.empty() ? string("-") : s;
  };

  if (!facts_face) {
    // Shadows the outer one for this face only: every row here steps by row_h.
    auto line = [&](const string& s) {
      draw_text(context, font, s, tx, ty, scale); ty += row_h;
    };
    // A boxed word the mouse can press, in the header buttons' clothes.
    auto button = [&](const string& label, int x, int y, ui_action act) {
      int bw = text_w(label, scale) + pad * 2;
      draw_boxed_text(context, font, label, x, y, scale, bw, box_h,
                      app_state.active_ui_action == act ? 4 : 1);
      app_state.click_zones.push_back({x, y, bw, box_h, act});
      return bw;
    };
    // Every control row reads NAME, what it is now, then the control. The name
    // column and the value field are fixed widths that don't depend on what they
    // say, so every control begins at the same place and none of them moves. The
    // value field holds only "WHICH OF HOW MANY", the widest of which is a track
    // saying OFF or ALL of some hundreds. What the split IS gets a line of its
    // own below, because saying it properly takes a sentence and a sentence
    // would push every control off the panel.
    const int value_chars = (int)string("OFF OF 9999").size();
    // EVERY LABEL THE PANE USES, so the value column is as wide as the widest
    // of them. This was the width of "DROPPED" alone, and the moment a label
    // grew past it the name ran straight through its own value.
    static const char* const pane_labels[] = {
        "READING", "LINE", "LEFT OVER", "REFUSED", "MOTIF", "EACH CELL" };
    int label_col = 0;
    for (const char* name : pane_labels) label_col = max(label_col, text_w(name, scale));
    // THREE CHARACTERS CLEAR OF THE LONGEST LABEL. At one character the widest
    // of them sat right against its own value and read as one word.
    const int value_x = tx + label_col + pad * 6;
    const int control_x = value_x + text_w(string(value_chars, ' '), scale) + pad * 6;
    // `chosen` inverts the value — black on white — for the reading inference
    // actually decided on. Two readings can score the same to the last cell, so
    // which one leads has to be visible, not remembered from an ordering.
    auto label_chip = [&](const string& name, int y, int x = -1) {
      if (x < 0) x = tx;
      if (!colored_fonts.count(0)) { draw_text(context, font, name, x, y, scale); return; }
      fill_rect(context, UI_COLOR_LINE, x - pad / 2, y - pad / 2,
                text_w(name, scale) + pad, 5 * scale + pad);
      draw_text(context, colored_fonts[0], name, x, y, scale);
    };
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

    // NO RULE ABOVE THE FIRST BLOCK — a divider parts two things, and there is
    // nothing above this one to part it from.
    // Which reading of the split is on screen, and how many there are. THE
    // NUMBER IS ITS CELL IN THE TABLE BELOW, counted across, so a step is a step
    // along that table and never a jump off it.
    // COUNTED FROM ZERO, like the log and like the lists themselves. Counted
    // from one it read as a different number from the same thing, and every
    // sentence about a reading had to say which of the two it meant.
    row_head("READING", fmt::format("{:>4} OF {}", rank, max(1, readings)),
             rank == task.correspondence.chosen);
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
      // Counted from zero, the same as READING above and the same as the log.
      row_head(name, at == SHOW_OFF ? fmt::format("{:>4} OF {}", "OFF", count)
                   : at == SHOW_ALL ? fmt::format("{:>4} OF {}", "ALL", count)
                                    : fmt::format("{:>4} OF {}", at, count));
      int x = control_x + button(fmt::format("< [SHIFT {}]", key), control_x, ty, back) + pad;
      x += button(fmt::format("> [{}]", key), x, ty, fwd) + pad;
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
    const auto& R = task.correspondence.readings[rank];
    // WHICH READING THIS IS, in words. The table below says it too, but only by
    // where the boxed cell sits, and hunting for it is work the reader should
    // not have to do to know what is on screen.
    detail(fmt::format("{} {}  ---  {}", split_names_lead,
                       split_names[clamp((int)R.cut_kind, 0,
                                         (int)split_kind::split_count - 1)],
                       match_mode::from_index(R.match_kind).label()));
    static const char* kind_name[] = { "EXACT", "RECOLORED", "SITS ON" };
    // The pieces the reading cut out, and the matches it threw away. Neither is
    // reachable through the lines: a piece with no edge draws nothing, and a
    // thrown-away match was never drawn at all.
    const vector<int> alone = pieces_standing_alone(R);
    const int pieces_n = (int)alone.size();
    const int dropped_n = (int)R.dropped_seen.size();
    const int left_over_n = (int)R.left_over_seen.size();
    track_row("LINE", "S", step, drawn, (int)arc_color::yellow,
              ui_action::step_edge_back, ui_action::step_edge, &app_state.inference_edge);
    // THE ONE LINE YOU ARE ON, said in full. All of this is invisible in the
    // drawing: a line looks the same whether the piece was found as itself,
    // found in another color, or never found at all.
    {
      string said;
      int nth = -1;
      for (const auto& ln : task.correspondence.lines) {
        if (ln.rank != rank) continue;
        if (++nth != step) continue;
        said = fmt::format("PAIR {}  COLOR {}  {}X{} AT {},{} -> {},{}  {}",
                           ln.pair, ln.color < 0 ? string("MIXED") : to_string(ln.color),
                           ln.from_w, ln.from_h, ln.from_x, ln.from_y,
                           ln.to_x, ln.to_y, kind_name[clamp((int)ln.kind, 0, 2)]);
        if (ln.pose & 1) said += ", TURNED";
        if (ln.pose & 2) said += ", FLIPPED ROWS";
        if (ln.pose & 4) said += ", FLIPPED COLS";
        if (ln.hidden)   said += fmt::format(", {} HIDDEN", ln.hidden);
        break;
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
        const auto& dr = R.dropped_seen[dropped_at];
        said = fmt::format("PAIR {}  PIECE {}  WANTED {},{}  {}  {}",
                           dr.pair, dr.piece, dr.to_x, dr.to_y,
                           kind_name[clamp((int)dr.kind, 0, 2)],
                           correspondence_facts::refusal_names[clamp((int)dr.why, 0, 2)]);
        if (dr.hidden) said += fmt::format(", {} HIDDEN", dr.hidden);
      }
      detail(said);
    }
    const vector<motif_set> sets = motif_sets(R);
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
        for (const auto& pc : R.pieces_seen) {
          if (!piece_is_in(pc, set)) continue;
          ++members;
          if (!pc.medoid) continue;
          leader_w = pc.w; leader_h = pc.h; leader_x = pc.x; leader_y = pc.y;
        }
        said = fmt::format("PAIR {} {}, {} COPIES OF {}X{}, LEADER AT {},{}",
                           set.pair, set.from_input ? "INPUT" : "OUTPUT",
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
        const auto& pc = R.pieces_seen[alone[piece_at]];
        const char* silence = pc.edges > 0 ? "ONLY SITS ON SOMETHING"
                            : pc.from_input ? "DELETED, IT IS NOT IN THE OUTPUT"
                                            : "INSERTED, IT WAS NOT IN THE INPUT";
        piece_said = fmt::format("{}  PAIR {}  COLOR {}  {}X{} AT {},{}  {} CELLS",
                                 silence, pc.pair,
                                 pc.color < 0 ? string("MIXED") : to_string(pc.color),
                                 pc.w, pc.h, pc.x, pc.y, pc.cells);
        if (pc.blank_w > 0)
          blank_said = fmt::format("{} {}X{} AT {},{}  {} CELLS",
                                   pc.blank_enclosed ? "FILLS A HOLE"
                                                     : "OVER OPEN SPACE",
                                   pc.blank_w, pc.blank_h, pc.blank_x, pc.blank_y,
                                   pc.blank_cells);
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
        const auto& cell = R.left_over_seen[left_over_at];
        said = fmt::format("PAIR {}  {}  CELL AT {},{}", cell.pair,
                           cell.from_input ? "INPUT" : "OUTPUT", cell.x, cell.y);
      }
      detail(said);
    }
    // A CONTROL ROW IS TALLER THAN A TEXT ROW, so it leaves more behind it than
    // divide() expects, and the first rule sat lower in its gap than the rest.
    // Give back the difference and all the gaps draw the same.

    // What this reading COST, the whole reason it leads or trails: pieces it
    // used and output cells no edge of any kind accounts for, over the train
    // set. Without it the coarseness steps are unnamed preferences.
    using readout = puzzle::correspondence_readout;
    // EVERY READING AT ONCE: how the grid was cut and which way round down
    // the side, what the pieces were allowed to do across the top. Stepping
    // them one at a time never says how they relate — the table says it in one
    // look, and the number in a cell is what that correspondence could not
    // account for ON EITHER GRID, so the best cells are the small ones and you
    // read for them without being told.
    divide();
    {
      const int cuts = (int)split_kind::split_count;
      // THE COLUMNS ARE WHAT FITS ACROSS. Freedoms doubled when hanging off the
      // edge became one of them, and sixteen columns ran off the panel and off
      // the screen. So the table says the other three across the top, as it
      // always did, and says this one by drawing the whole block of cuts twice:
      // once for pieces that must stay on the grid, once for pieces that may
      // hang off it. Same reading in the same place either way — a cell is
      // still its reading's own cell, the block just says which half it is in.
      const int modes = match_mode::COUNT / 2;
      // THE LIST IS THIS TABLE. A reading is kept at its own cell, one row per
      // cut and one column per freedom, so a cell knows which reading it stands
      // for without anything having to be looked up — and stepping the reading
      // walks the table you are looking at.
      // A ROW IS NAMED BY WHAT IT COST TO LOOK AT, because that is what decided
      // whether it got searched at all. Said in the label, not in a column of
      // its own: it is one number per row, and a column would repeat it eight
      // times as though the freedoms changed it.
      // NO COST COLUMN. What a cut costs to say orders the walk and breaks a
      // dead tie, and nothing else — the rows here are in cut order, not cost
      // order, so a number beside them read as a ranking that ranks nothing.
      // It is still in the log, where a diagnostic belongs.
      int label_w = text_w(split_names_lead, scale);
      for (int c = 0; c < cuts; ++c)
        label_w = max(label_w, text_w(split_names[c], scale));
      // EVERY CELL IS THE SAME BOX, as wide as the widest thing a column ever
      // holds — a freedom's name, or the biggest number. Cut to its own text a
      // single digit read as a smaller thing than three, and the boxes down a
      // column no longer lined up with each other.
      // MEASURED OVER WHAT A COLUMN ACTUALLY HOLDS. Measured over every freedom
      // name, the one that has no column — it titles the second block instead —
      // still set the width of all of them, and the table ran off the panel.
      int widest = text_w("9999", scale);
      for (int freedom = 0; freedom < match_mode::FREEDOMS - 1; ++freedom)
        widest = max(widest, text_w(match_mode::freedom_names[freedom], scale));
      const int cell_w = widest + pad * 2;
      const int chip_w = widest + pad;
      // THE LABEL ENDS IN A NUMBER, so a one-character gap made that number read
      // as the first cell of the row. The columns start a clear space away.
      const int gx = tx + label_w + pad * 10;
      // EACH COLUMN SAYS WHAT IT IS, AT THE COLUMN. Three named rows of bars
      // meant reading downwards to decode one column, and a freedom that was
      // OFF drew nothing — which looks the same as a row that is missing.
      // A dash is present and legible; a blank is not.
      // TWO AXES, TWO TITLES, AND THEY FINISH ONE SENTENCE. Down the side is
      // what an object IS; across the top is what it MAY DO. A reader who has
      // both can say a cell out loud without a legend: "an object is a group of
      // cells that touch, and it may move and turn".
      // CENTRED ON WHAT THEY TITLE. A heading pinned to the left edge of a wide
      // block reads as the first item of it, not as the name of the whole.
      auto centred = [&](const string& name, int from_x, int width) {
        label_chip(name, ty, from_x + (width - text_w(name, scale)) / 2);
      };
      // EVERY COLUMN IS THE SAME BOX WIDE, so its heading sits in the middle of
      // it and so does what it holds. Pinned to the left edge instead, a number
      // drifts as it gains a digit and the heading no longer names the column
      // under it.
      auto cell_x = [&](int m) { return gx + m * cell_w - pad / 2; };
      // ONE BOX FOR A CELL AND FOR THE HEADING OVER IT. Cut to its own text
      // instead, a three-letter heading and a one-digit number were two widths
      // in the same column. Hands back where it drew, so a cell that wants a
      // border round it puts it on the box and not on the writing.
      auto column_chip = [&](const string& s, int m, bool inverted) {
        const int chip_x = cell_x(m) + (cell_w - chip_w) / 2;
        if (inverted && colored_fonts.count(0))
          fill_rect(context, UI_COLOR_LINE, chip_x, ty - pad / 2, chip_w, 5 * scale + pad);
        draw_text(context, inverted && colored_fonts.count(0) ? colored_fonts[0] : font,
                  s, cell_x(m) + (cell_w - text_w(s, scale)) / 2, ty, scale);
        return chip_x;
      };
      centred("AND IT MAY", cell_x(0), modes * cell_w);
      ty += row_h;
      // ONE FREEDOM PER LINE, IN ITS OWN WORD. Three initials side by side
      // needed a legend under the table to be read at all, and one of them
      // could be taken for either of two freedoms. A dash where a word would
      // be says the freedom was withheld — a blank would look like a gap.
      // The freedoms that have a column of their own: everything but the last.
      const int across = match_mode::FREEDOMS - 1;
      for (int freedom = 0; freedom < across; ++freedom) {
        // On the last line, so it sits against the sentences it names.
        if (freedom == across - 1)
          centred(split_names_lead, tx, label_w);
        for (int m = 0; m < modes; ++m)
          column_chip(match_mode::from_index(m).may(freedom)
                          ? match_mode::freedom_names[freedom] : "-", m, true);
        ty += row_h;
      }
      // THE SAME ROWS, TWICE, THE SECOND TIME WITH THE LAST FREEDOM GRANTED.
      // Its own name titles the second block, so the sentence still finishes:
      // an object is a group of cells that touch, and it may move and turn and
      // hang off the edge.
      for (int half = 0; half < 2; ++half) {
        if (half) {
          centred(match_mode::freedom_names[across], cell_x(0), modes * cell_w);
          ty += row_h;
        }
        for (int c = 0; c < cuts; ++c) {
          label_chip(split_names[c], ty);
          for (int m = 0; m < modes; ++m) {
            const int i = c * match_mode::COUNT + m + half * modes;
            if (i >= readings) continue;
            const string s = to_string(task.correspondence.readings[i].left_over);
            // The one inference chose reads inverted; the one on screen is boxed.
            const int chip_x = column_chip(s, m, i == task.correspondence.chosen);
            if (i == rank)
              draw_border(context, chip_x, ty - pad / 2, chip_w, 5 * scale + pad,
                          max(1, scale / 2));
            app_state.pick_zones.push_back({cell_x(m), ty - (row_h - 5 * scale) / 2,
                                            cell_w, row_h, &app_state.inference_rank.at, i});
          }
          ty += row_h;
        }
      }
      named_line("EACH CELL", fmt::format(
          "CELLS IT COULD NOT ACCOUNT FOR, OF {} ON BOTH GRIDS, EVERY PAIR",
          task.correspondence.cells));
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
    for (size_t i = 0; i < R.pairs.size(); ++i) {
      int kinds[3] = {}, hidden = 0, turned = 0;
      for (const auto& ln : task.correspondence.lines) {
        if (ln.rank != rank || ln.pair != (int8_t)i) continue;
        kinds[clamp((int)ln.kind, 0, 2)]++;
        hidden += (ln.hidden > 0);
        turned += (ln.pose != 0);
      }
      // Places one piece went is an input-side question, pieces meeting at one
      // landing an output-side one. Both ends are read, so both are counted.
      int most_landings = 0, most_sources = 0;
      for (const auto& pc : R.pieces_seen) {
        if (pc.pair != (int8_t)i) continue;
        int& most = pc.from_input ? most_landings : most_sources;
        most = max(most, (int)pc.edges);
      }
      const auto& pr = R.pairs[i];
      pair_row(to_string(i), how_name[clamp((int)pr.how, 0, 2)],
               fmt::format("{}-{}", most_sources, most_landings),
               to_string(kinds[0]), to_string(kinds[1]), to_string(kinds[2]),
               to_string(hidden), to_string(turned),
               to_string(pr.dropped), to_string(pr.left_over));
    }
    // Why the dropped ones were dropped, over the whole reading.
    {
      int by_reason[3] = {};
      for (const auto& pr : R.pairs) {
        by_reason[0] += pr.chance; by_reason[1] += pr.stale; by_reason[2] += pr.did_not_fit;
      }
      // ONE REASON A LINE. All three on one line came to more characters than
      // the panel is wide, and the widest screen is the tighter of the two:
      // the label column grows with the font while the panel grows with the
      // screen. The label is said once and the rest sit under it.
      for (int reason = 0; reason < 3; ++reason) {
        if (reason == 0) label_chip("REFUSED", ty);
        draw_text(context, font,
                  fmt::format("{} {}", by_reason[reason],
                              correspondence_facts::refusal_names[reason]),
                  value_x, ty, scale);
        ty += line_h;
      }
    }
    return;
  }

  {
    // Size · what decides how big the output is.
    static const char* rule_name[] = { "NOT INFERRED", "SAME AS INPUT", "FIXED",
                                       "SCALED", "WHAT IT DRAWS" };
    string s = fmt::format("OUTPUT SIZE  {}", rule_name[(int)task.rule.Chosen]);
    if (task.rule.Chosen == output_dimension_rule::Fixed)
      s += fmt::format("  {}X{}", task.rule.width, task.rule.height);
    if (task.rule.Chosen == output_dimension_rule::Scale)
      s += fmt::format("  {}/{} BY {}/{}", task.rule.width_numerator, task.rule.width_denominator,
                       task.rule.height_numerator, task.rule.height_denominator);
    line(s);
  }
  line(fmt::format("COLORS GONE {}   NEW {}   KEPT {}",
                   color_list(F.consumed), color_list(F.introduced), color_list(F.preserved)));
  line(fmt::format("IN EVERY INPUT {}   IN EVERY OUTPUT {}",
                   color_list(F.every_input), color_list(F.every_output)));
  line(fmt::format("IN EVERY GRID {}   ANYWHERE {}",
                   color_list(F.every_grid), color_list(F.any_grid)));
  {
    string s = "OUTPUT COLORS ";
    for (int i = 0; i < task.output_colors_count; ++i)
      s += fmt::format("{}", (int)task.output_colors[i]);
    if (task.output_colors_by_size_count) {
      s += "   BY SIZE ";
      for (int i = 0; i < task.output_colors_by_size_count; ++i)
        s += fmt::format("{}", (int)task.output_colors_by_size[i]);
    }
    line(s);
  }
  ty += line_h;
  {
    string head = fmt::format("BG INPUT {}  OUTPUT {}",
                              (int)bg.input_elected, (int)bg.output_elected);
    if (bg.per_pair_mode) head += "  MODE PER-PAIR";
    draw_text(context, font, head, tx, ty, scale);
  }
  ty += line_h;
  {
    string tier = fmt::format("VETOED {}", bg.input_vetoed ? "YES" : "NO");
    draw_text(context, font, tier, tx, ty, scale);
    ty += line_h;
  }
  {
    string pp = "PER PAIR";
    for (int8_t c : bg.per_pair) pp += fmt::format(" {}", (int)c);
    draw_text(context, font, pp, tx, ty, scale);
    ty += line_h + line_h;  // blank line after
  }

  auto draw_dir = [&](const char* name, const vector<puzzle::background_readout::cand>& d) {
    draw_text(context, font, name, tx, ty, scale); ty += line_h;
    // column header, aligned with the rows below
    draw_text(context, font, fmt::format("{:>7} {:>10} {:>10} {:>8} {:>7} {:>6} {:>6} {:>6}",
              "COLOR", "EMBEDS", "DOMINANCE", "PERSISTS", "SCORE", "SPAN", "GAPS", "UNDER"),
              tx + sw + scale, ty, scale);
    ty += line_h;
    for (const auto& c : d) {
      swatch(c.color, tx, ty);
      string label = (c.color < 0) ? "NONE" : fmt::format("{}", (int)c.color);
      string row = fmt::format("{:>7} {:>10} {:>10} {:>8} {:>7} {:>6} {:>6} {:>6}",
                               label, c.embeds, c.dominance, c.persists ? "YES" : "NO", c.score, c.span, c.gaps,
                               c.underneath ? "YES" : "NO");
      draw_text(context, font, row, tx + sw + scale, ty, scale);
      ty += line_h;
    }
    ty += line_h / 2;   // gap between elections
  };
  draw_dir("ACROSS INPUTS", bg.across_inputs);
  draw_dir("ACROSS OUTPUTS", bg.across_outputs);
  draw_dir("ACROSS IN+OUT", bg.across_inputs_outputs);

  // What changed between the two sides.
  ty += line_h / 2;
  if (F.unchanged_valid)
    line(fmt::format("INPUT ALREADY RIGHT ON {:.0f} PCT OF CELLS, WORST PAIR",
                     F.min_kept * 100.0f));
  if (F.diff_valid) {
    string s = "CELLS WON/LOST ";
    for (int c = 0; c < 10; ++c)
      if (F.gained[c] || F.lost[c])
        s += fmt::format(" {}:{}/{}", c, F.gained[c], F.lost[c]);
    line(s);
  }
  if (task.color_swap_variants_count) {
    string s = "SWAPPED";
    int k = 0;
    for (int a = 0; a < 10; ++a)
      for (int b = a + 1; b < 10; ++b, ++k)
        if (task.color_swap_variants[k]) s += fmt::format(" {}<>{}", a, b);
    line(s);
  }
  if (task.shape_color_count)
    line(fmt::format("SHAPE PICKS COLOR, {} LEARNED", task.shape_color_count));

  // What correspondence made of each pair, said FROM THE INPUT'S SIDE, which is
  // the same standpoint the ALONE row uses. DELETED and INSERTED are one fact
  // read from two ends — no piece on the other grid is this one — and which end
  // is just which grid it sits on, so the two must not read as two different
  // findings. CLASS is the two extremes the six counts hide: most pieces
  // meeting at one landing, most places one piece went. 1-1 is a plain swap,
  // 1-9 is one piece stamped nine times, 3-2 is both at once.
  ty += line_h / 2;
  const char* pair_columns_wide =
      "{:>4} {:>9} {:>5} {:>6} {:>9} {:>8} {:>6} {:>8} {:>8} {:>5}";
  draw_text(context, font,
            fmt::format("{:>4} {:^22} {:^25} {:>8} {:>6}",
                        "", "EACH INPUT PIECE", "EACH OUTPUT PIECE", "", ""),
            tx, ty, scale);
  ty += line_h;
  draw_text(context, font,
            fmt::format(fmt::runtime(pair_columns_wide),
                        "PAIR", "DELETED", "KEPT", "COPIED",
                        "INSERTED", "FROM ONE", "MERGED", "CLASS", "CARRIED", "MADE"),
            tx, ty, scale);
  ty += line_h;
  for (size_t i = 0; i < F.pairs.size(); ++i) {
    const auto& p = F.pairs[i];
    line(fmt::format(fmt::runtime(pair_columns_wide),
                     i, p.deleted, p.persisted, p.copied,
                     p.invented, p.from_one, p.from_many,
                     fmt::format("{}-{}", p.most_sources, p.most_landings),
                     p.carried_cells, p.created_cells));
  }

}

inline void draw_op_palette(vulkan_context *context, const map<uint32_t, grid> &font,
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
  auto labels_for = [&](int tab) {
    vector<int> ids;
    for (int i = 0; i < n; ++i)
      if ((int)role_of((node_type)i) == tab) ids.push_back(i);
    sort(ids.begin(), ids.end(), [](int a, int b) {
      return string(type_name[a]) < string(type_name[b]);
    });
    vector<string> labels;
    for (int i : ids) {
      string s = type_name[i];
      if (is_annotated((node_type)i)) s += fmt::format("[0-{}]", type_var_count[i] - 1);
      labels.push_back(s);
    }
    return labels;
  };

  vector<string> part_labels;
  for (int r = 0; r < part_count; ++r) part_labels.push_back(sentence[r].name);
  // Sub-kind row: only drawn when the active role has more than one kind.
  bool has_subrow = sentence[active_part].kinds.size() > 1;
  vector<string> sub_labels;
  if (has_subrow) for (op_role k : sentence[active_part].kinds) sub_labels.push_back(tab_names[(int)k]);
  vector<string> op_labels = labels_for(active);

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
    int t = drawer_border_t(scale), off = solid_sheets[UI_COLOR_LINE].offset;
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
    context->draw(solid_sheets[5].offset, lay.screen_w - 2 * hmargin, 1,
                  hmargin, sep_y, 1, 1);
  int list_top = sep_y + vgap;
  int avail = lay.screen_w - 2 * hmargin;
  // Group ops into rows first, so each row can be horizontally centered.
  vector<vector<pair<string,int>>> rows(1); // (label, box_w)
  int row_w = 0;
  for (const string &label : op_labels) {
    int bw = text_w(label, scale) + pad * 2;
    int add = bw + (rows.back().empty() ? 0 : pad);
    if (row_w + add > avail && !rows.back().empty()) { rows.emplace_back(); row_w = 0; add = bw; }
    rows.back().push_back({label, bw});
    row_w += add;
  }
  int y = list_top;
  for (auto &row : rows) {
    int total = 0;
    for (auto &c : row) total += c.second;
    total += (int)(row.size() - 1) * pad;
    int x = max(hmargin, (lay.screen_w - total) / 2);
    for (auto &c : row) {
      if (y >= 0)
        draw_boxed_text(context, font, c.first, x, y, scale, c.second, box_h, 1);
      x += c.second + pad;
    }
    y += line_h;
  }
}

// Main frame draw - orchestrates header and content area
inline void draw_frame(vulkan_context *context, const map<uint32_t, grid> &font,
                       const grid &background, const solver_state &state,
                       global_app_state &app_state, const puzzle &task,
                       int screen_w, int screen_h, float fps, float aps, float gpu_ms,
                       int puzzle_idx, int puzzle_count,
                       vector<int> &gpu_buffer) {

  frame_layout lay = compute_frame_layout(screen_w, screen_h);

  long long current_attempts = state.attempts.load();
  int search_depth_limit = state.current_max_depth;
  int best_depth = task.best_tree ? task.best_tree->max_depth : 0;
  int cur_depth = state.current_attempt_depth.load();

  float elapsed = get_elapsed(app_state, state);

  app_state.click_zones.clear();
  app_state.slider_zones.clear();
  app_state.pick_zones.clear();
  {
    lock_guard<mutex> lk(app_state.node_zones_mtx);
    app_state.node_zones.clear();

    // The header takes as many lines as its pieces need, so everything under it
    // is placed from where it actually ended, not from a reserved row count.
    lay.grid_labels_y = draw_header_rows(context, font, app_state, lay,
                                         task, puzzle_idx, puzzle_count,
                                         current_attempts, aps, fps, gpu_ms,
                                         cur_depth, best_depth, search_depth_limit);
    int grid_label_h = 5 * lay.scale + 2 * 2 * lay.scale; // matches label box height
    lay.grid_data_start_y = lay.grid_labels_y + grid_label_h + lay.v_gap;

    // Reflow the BEST/CURRENT footers for multi-line (wrapped) pipelines: deep
    // programs stack upward (and shrink the grid area) instead of running off the
    // right edge. Single-line programs reproduce the original footer positions
    // exactly (cur_lines == best_lines == 1 -> footer{2,1}_y a row_h apart).
    {
      int fscale = lay.scale, fpad = 2 * fscale;
      int max_x = lay.screen_w - lay.base_margin;
      int line_h = 9 * fscale + lay.v_gap;
      int best_lw = text_w("BEST", fscale) + fpad * 2;
      int cur_lw = text_w("CURRENT", fscale) + fpad * 2;
      // Always reserve at least 3 lines each so the layout doesn't jump on wrap.
      int best_lines = max(3, task.best_tree
          ? count_pipeline_lines(task.best_tree, lay.base_margin + best_lw + fpad, max_x, fscale, fpad)
          : 1);
      int cur_lines = max(3, state.current_tree
          ? count_pipeline_lines(state.current_tree, lay.base_margin + cur_lw + fpad, max_x, fscale, fpad)
          : 1);
      lay.footer2_y = lay.footer3_y - best_lines * line_h;
      lay.footer1_y = lay.footer2_y - cur_lines * line_h;
      // Safety: never let footers climb into the grid-label band (absurdly deep
      // programs); the grid just shrinks to whatever room remains.
      if (lay.footer1_y < lay.grid_data_start_y + line_h)
        lay.footer1_y = lay.grid_data_start_y + line_h;
    }

    draw_footer_rows(context, font, app_state, lay, task, state, elapsed);
  }

  if (!draw_grid_view(context, font, background, task, lay))
    return;

  // Hover preview: compute & draw the subtree output for every train+test
  // example, overlaid on top of the corresponding row in the CURRENT column.
  {
    int p_idx = state.puzzle_idx.load();
    if (state.preview_last_puzzle_idx != p_idx) {
      state.preview_cache.clear();
      state.preview_uploaded_hash = 0;
      state.preview_last_puzzle_idx = p_idx;
    }

    uint64_t h = app_state.hovered_hash.load();
    int hs = app_state.hovered_source.load();
    node *root = (hs == 1) ? task.best_tree : state.current_tree;
    int total_examples = (int)(task.train.size() + task.test.size());
    if (h != 0 && root && total_examples > 0) {
      auto it = state.preview_cache.find(h);
      if (it == state.preview_cache.end()) {
        if (node *target = find_node_by_hash(root, h)) {
          int nc = count_subtree_nodes(target);
          solver_state::preview_payload payload;
          payload.accumulators.resize(total_examples);
          payload.object_lists.resize(total_examples);
          payload.object_bboxes.resize(total_examples);
          int slot = 0;
          auto run_section = [&](const vector<example> &vec) {
            for (const auto &ex : vec) {
              grid out;
              float score = 0; bool solved = false;
              bool ok = environment::execute_tree_via_flat(
                  target, ex, out, task, nc, score, solved);
              if (ok) payload.accumulators[slot] = out;

              // Pull whatever object list the hovered op produced (if any).
              // Canvas must be large enough to hold every object's painted
              // extent â€” input-sized isn't enough when ops like Fractal/Tile
              // produce output-sized masks. Use max(input, output, every
              // object's x1+1/y1+1) so test examples (output dims = 0) and
              // multi-scale ops both display correctly.
              gpu_object_list ol;
              int work_w = 0, work_h = 0;
              if (cuda_dsl_fetch_last_object_list_scratch(0, ol, &work_w, &work_h)) {
                int n_obj = min((int)ol.count, GPU_KERNEL_MAX_OBJECTS);
                payload.object_lists[slot].reserve(n_obj);
                payload.object_bboxes[slot].reserve(n_obj);
                // Base the tile canvas on the job's working output canvas, which
                // reflects rotate ops (RotateToPortrait yields a portrait canvas) so
                // the background rotates with the objects. Fall back to the example
                // dims when the kernel reported nothing; grow to fit any object that
                // paints beyond the canvas either way.
                int canvas_w = (work_w > 0) ? work_w : max((int)ex.input.width,  (int)ex.output.width);
                int canvas_h = (work_h > 0) ? work_h : max((int)ex.input.height, (int)ex.output.height);
                for (int k = 0; k < n_obj; ++k) {
                  const auto &o = ol.objects[k];
                  if (o.x1 + 1 > canvas_w) canvas_w = o.x1 + 1;
                  if (o.y1 + 1 > canvas_h) canvas_h = o.y1 + 1;
                }
                canvas_w = min(canvas_w, MAX_GRID_DIM);
                canvas_h = min(canvas_h, MAX_GRID_DIM);
                for (int k = 0; k < n_obj; ++k) {
                  const auto &o = ol.objects[k];
                  grid g;
                  g.width = canvas_w;
                  g.height = canvas_h;
                  // Default-fill canvas with transparent; only paint mask
                  // cells that are themselves non-transparent.
                  for (int r = 0; r < canvas_h; ++r)
                    for (int c = 0; c < canvas_w; ++c)
                      g.data[r][c] = GPU_TRANSPARENT;
                  int mh = min((int)o.height, MAX_GRID_DIM);
                  int mw = min((int)o.width, MAX_GRID_DIM);
                  for (int r = 0; r < mh; ++r) {
                    int dst_r = o.y0 + r;
                    if (dst_r < 0 || dst_r >= canvas_h) continue;
                    for (int c = 0; c < mw; ++c) {
                      int dst_c = o.x0 + c;
                      if (dst_c < 0 || dst_c >= canvas_w) continue;
                      int8_t v = o.mask[r][c];
                      if (v != GPU_TRANSPARENT)
                        g.data[dst_r][dst_c] = (int)v;
                    }
                  }
                  payload.object_lists[slot].push_back(std::move(g));
                  payload.object_bboxes[slot].push_back({
                    (int)o.x0, (int)o.y0, (int)o.x1, (int)o.y1, canvas_w, canvas_h});
                }
              }
              ++slot;
            }
          };
          run_section(task.train);
          run_section(task.test);
          it = state.preview_cache.emplace(h, std::move(payload)).first;
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
        int available_grid_h = lay.footer1_y - lay.grid_data_start_y;
        int slot_h = available_grid_h / total_examples;
        int col_w = lay.screen_w / 4;
        int outline = lay.scale;
        int n = min((int)state.preview_grids.size(), total_examples);

        // Backdrop spans CURRENT (col 2) + BEST (col 3) for every example row,
        // giving the preview ~half the screen width to lay out accumulator +
        // object strip. It also extends upward to cover the BEST/CURRENT
        // header labels so the preview takes over the whole right half.
        int back_x = 2 * col_w + lay.base_margin;
        int back_w = 2 * col_w - lay.base_margin * 2;

        // Cover headers once, above the first row, and place a single
        // "PREVIEW" label centered over the doubled column.
        {
          int hdr_y = lay.grid_labels_y;
          int hdr_h = lay.grid_data_start_y - lay.grid_labels_y;
          const grid &back_sheet = solid_sheets.count(0) ? solid_sheets[0] : background;
          // stride=1, not the sheet's real stride: the shader reads
          // cells[offset + row*stride + col], so a fill wider than the sheet
          // (back_w ~1256 vs the 256-wide solid sheet) walks past the block
          // into the next color sheet (blue) and leaks a band â€” visible once
          // the area is big enough (2K, not 1080). With stride=1 every cell
          // reads offset+row+col, always inside the solid block; positions are
          // computed independently so the rectangle still fills correctly.
          context->draw(back_sheet.offset, back_w, hdr_h,
                        back_x, hdr_y, 1, 1);

          int grid_pad = 2 * lay.scale;
          int lw = text_w("PREVIEW", lay.scale) + grid_pad * 2;
          int lh = 5 * lay.scale + grid_pad * 2;
          // Flush-left at the CURRENT column's label x (col index 2).
          int label_x = 2 * col_w + lay.base_margin;
          draw_boxed_text(context, font, "PREVIEW", label_x, lay.grid_labels_y,
                          lay.scale, lw, lh, 5);
        }

        for (int i = 0; i < n; ++i) {
          const grid &m = state.preview_grids[i];
          int row_y = lay.grid_data_start_y + i * slot_h;
          int back_y = row_y + pad;
          int back_h = slot_h - pad * 2;

          // Black backdrop covers CURRENT + BEST for this row. Use color 0
          // (black in the ARC palette) rather than the cell-outline grid,
          // which is a non-black UI color.
          const grid &back_sheet = solid_sheets.count(0) ? solid_sheets[0] : background;
          // stride=1: solid fill wider than the 256-wide sheet would otherwise
          // overrun into the next color sheet and leak a blue band (see header
          // backdrop above).
          context->draw(back_sheet.offset, back_w, back_h,
                        back_x, back_y, 1, 1);

          int x_inner = back_x + outline;
          int y_top = back_y + outline;
          int inner_w = back_w - outline * 2;
          int y_used = y_top;

          // 1) Accumulator preview on top of the backdrop, with the same grey
          // outline frame the regular grid cells use.
          if (m.width > 0 && m.height > 0) {
            string dim = fmt::format("{:02d}x{:02d}", m.width, m.height);
            int dim_w = text_w(dim, lay.scale) + outline * 2;
            int half_h = (back_h - outline * 2) / 2;
            int cell_scale = min((inner_w - dim_w) / m.width, half_h / m.height);
            if (cell_scale < 1) cell_scale = 1;
            int gw = m.width * cell_scale, gh = m.height * cell_scale;
            context->draw(background.offset, gw + outline * 2, gh + outline * 2,
                          x_inner - outline, y_top - outline, 1, background.stride);
            context->draw(m.offset, m.width, m.height, x_inner, y_top, cell_scale, m.stride);
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
            for (const grid &og : slots) {
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
              const grid &line_sheet = solid_sheets.count(UI_COLOR_LINE)
                                          ? solid_sheets[UI_COLOR_LINE]
                                          : background;
              int line_th = max(1, lay.scale / 2);
              for (size_t bi = 0; bi < slots.size(); ++bi) {
                const grid &og = slots[bi];
                if (og.width <= 0 || og.height <= 0) continue;
                int gw = og.width * cell_scale, gh = og.height * cell_scale;
                context->draw(background.offset, gw + outline * 2, gh + outline * 2,
                              x_cur - outline, y_used - outline, 1, background.stride);
                context->draw(og.offset, og.width, og.height, x_cur, y_used, cell_scale, og.stride);
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
                    context->draw(line_sheet.offset, bw, t,
                                  bx, by, 1, line_sheet.stride);
                    context->draw(line_sheet.offset, bw, t,
                                  bx, by + bh - t, 1, line_sheet.stride);
                    context->draw(line_sheet.offset, t, bh,
                                  bx, by, 1, line_sheet.stride);
                    context->draw(line_sheet.offset, t, bh,
                                  bx + bw - t, by, 1, line_sheet.stride);
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
    const int readings = max(1, (int)task.correspondence.readings.size());
    // Told which reading inference chose and how many there are, it hands back
    // the one to show — following the chosen one until somebody steps away.
    const int rank =
        app_state.inference_rank.pick(task.correspondence.chosen, readings);
    int drawn = 0;
    for (const auto &ln : task.correspondence.lines)
      if (ln.rank == rank) drawn++;
    // Each track is told how long its list is and hands back what is chosen.
    const auto &R = task.correspondence.readings;
    const vector<int> alone =
        rank < (int)R.size() ? pieces_standing_alone(R[rank]) : vector<int>{};
    const int pieces_n = (int)alone.size();
    const int dropped_n = rank < (int)R.size() ? (int)R[rank].dropped_seen.size() : 0;
    const int step = app_state.inference_edge.pick(drawn);
    const int piece_at = app_state.inference_piece.pick(pieces_n);
    const int dropped_at = app_state.inference_dropped.pick(dropped_n);
    const vector<motif_set> sets =
        rank < (int)R.size() ? motif_sets(R[rank]) : vector<motif_set>{};
    const int motif_at = app_state.inference_motif.pick((int)sets.size());
    const int left_over_n = rank < (int)R.size() ? (int)R[rank].left_over_seen.size() : 0;
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
        const auto &dr = R[rank].dropped_seen[i];
        const bool one = (dropped_at == i);
        // ONE shows the refused piece and where it wanted to go; ALL shows only
        // the wanted landings. So the first box is on the piece's own grid when
        // one is singled out and on the other grid otherwise.
        draw_selection(context, task, lay, dr.pair, one == dr.from_input,
                       one ? dr.from_x : dr.to_x, one ? dr.from_y : dr.to_y,
                       one ? dr.from_w : dr.to_w, one ? dr.from_h : dr.to_h,
                       one, dr.to_x, dr.to_y, dr.to_w, dr.to_h,
                       (int)arc_color::red, rings);
      }
      // COPIES OF ONE THING, ONE SET AT A TIME. A grid can hold several
      // motifs and one color cannot say which box belongs with which, so the
      // set is stepped like everything else. MAGENTA IS THE MEDOID and blue the
      // rest: a set wants an exemplar, and it cannot be the biggest or the
      // first — both pick by luck, and a block the noise landed on is complete
      // and still the odd one out.
      for (const auto &pc : R[rank].pieces_seen) {
        if (pc.motif < 0) continue;
        if (motif_at != SHOW_ALL &&
            !(motif_at < (int)sets.size() && piece_is_in(pc, sets[motif_at]))) continue;
        draw_selection(context, task, lay, pc.pair, pc.from_input,
                       pc.x, pc.y, pc.w, pc.h, false, 0, 0, 0, 0,
                       (int)(pc.medoid ? arc_color::magenta : arc_color::blue),
                       rings);
      }
      // A BOX AND NO LINE. These are the pieces nothing partners, so there is no
      // other end to join to — but there is a place: the patch of nothing they
      // cover on the other grid, drawn there so the two are seen together. That
      // patch is where a grid makes something out of nothing.
      for (int i = 0; i < pieces_n; ++i) {
        if (piece_at != SHOW_ALL && piece_at != i) continue;
        const auto &pc = R[rank].pieces_seen[alone[i]];
        draw_selection(context, task, lay, pc.pair, pc.from_input,
                       pc.x, pc.y, pc.w, pc.h, pc.blank_w > 0,
                       pc.blank_x, pc.blank_y, pc.blank_w, pc.blank_h,
                       (int)arc_color::gray, rings);
      }
      // THE CELLS NOTHING ACCOUNTED FOR, one cell each, so there is never room
      // to step inside another outline — they are drawn last and simply sit
      // where they are.
      for (int i = 0; i < left_over_n; ++i) {
        if (left_over_at != SHOW_ALL && left_over_at != i) continue;
        const auto &cell = R[rank].left_over_seen[i];
        draw_selection(context, task, lay, cell.pair, cell.from_input,
                       cell.x, cell.y, 1, 1, false, 0, 0, 0, 0,
                       (int)arc_color::orange, rings);
      }
    }
    draw_inference_pane(context, font, app_state, task, lay, view, rank, step,
                        drawn, piece_at, dropped_at, motif_at, left_over_at);
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

inline void load_font_atlas(vector<int> &gpu_buffer, map<uint32_t, grid> &chars) {
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
    grid g = create_character_grid(cp);
    g.offset = (int)gpu_buffer.size();
    g.stride = 4;
    for (int r = 0; r < 5; ++r) {
      for (int c = 0; c < 3; ++c)
        gpu_buffer.push_back(g.data[r][c]);
      gpu_buffer.push_back(0);
    }
    chars[cp] = g;
  }
  auto push_block = [&](int val, char ch) {
    pad_buffer(gpu_buffer);
    grid g{1, 1, (int)gpu_buffer.size(), 4};
    gpu_buffer.insert(gpu_buffer.end(), {val, 0, 0, 0});
    chars[ch] = g;
  };
  push_block(UI_COLOR_LINE, (char)255);
  push_block(0, (char)254);
  for (int i = 0; i < (int)arc_color::count; ++i)
    push_block(i, (char)(240 + i));

  vector<int> colors;
  colors.reserve(64);
  for (int i = 0; i < (int)arc_color::count; ++i)
    colors.push_back(i);
  for (int i = 0; i < (int)arc_color::count; ++i)
    colors.push_back(UI_COLOR_PASTEL_START + i);
  // Add Line Color
  colors.push_back(UI_COLOR_LINE);

  for (int c_idx : colors) {
    pad_buffer(gpu_buffer);
    solid_sheets[c_idx] = grid(256, 256, (int)gpu_buffer.size(), 256);
    gpu_buffer.insert(gpu_buffer.end(), 256 * 256, c_idx);
    for (auto const &[k, v] : chars) {
      if ((k < 32 || k > 126) && k != 0x25B2 && k != 0x25BC &&
          k != 0xE000 && k != 0xE001 && k != 0xE002 && k != 0xE003)
        continue;
      pad_buffer(gpu_buffer);
      grid cv = v;
      cv.offset = (int)gpu_buffer.size();
      cv.stride = 4;
      for (int r = 0; r < cv.height; ++r) {
        for (int c = 0; c < cv.width; ++c) {
          if (c_idx < UI_COLOR_PASTEL_START) {
            // Label Mode: Colored letters on transparent background
            gpu_buffer.push_back(v.data[r][c] == 10 ? c_idx : -1);
          } else {
            // Box Mode: White letters on colored background
            gpu_buffer.push_back(v.data[r][c] == 10 ? UI_COLOR_LINE : c_idx);
          }
        }
        gpu_buffer.push_back(0);
      }
      colored_fonts[c_idx][k] = cv;
    }
  }
}

}
