#pragma once
#include "28_global_app_state.hpp"
#include "33_os_utilities.hpp"
#include "34_solver_session.hpp"
#include "31_sound_manager.hpp"
#include <windows.h>
#include <cctype>
#include <cstdlib>   // std::getenv (ARC_W / ARC_H debug overrides)
#include <cstring>
#include <limits>

namespace arc {

// A developer knob read once at startup and never written back. The CRT calls
// getenv unsafe because a later putenv can invalidate the pointer it handed
// out, which cannot happen here -- so the exception is stated once, in the one
// place that takes it, rather than switched off for the whole build.
inline const char *debug_override(const char *name) {
#pragma warning(suppress : 4996)
  return std::getenv(name);
}

class window_manager;
inline window_manager *g_window_manager_ptr = nullptr;

class window_manager {
public:
  inline window_manager(global_app_state &app_state,
                        solver_session &solver_threading)
      : m_app_state(app_state), m_solver_threading(solver_threading) {
    g_window_manager_ptr = this;
  }

  inline void init(HINSTANCE h_instance, int n_show_cmd) {
    disable_power_throttling();
    disable_fullscreen_optimizations();
    // Per-Monitor-V2 keeps GetSystemMetrics, the swapchain, and WM_MOUSE*
    // coordinates all in true physical pixels on whatever monitor we land on.
    // The old SetProcessDPIAware() only requests *System* awareness, which
    // makes Windows bitmap-stretch the window when the panel's DPI differs
    // from the logon-DPI primary (e.g. 2K @ 150%). That stretch desyncs the
    // rendered op-token layout from the mouse hit-test, breaking the hover
    // preview. V2 (Win10 1703+) avoids the stretch; fall back if unavailable.
    if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
      SetProcessDPIAware();

    m_width = GetSystemMetrics(SM_CXSCREEN);
    m_height = GetSystemMetrics(SM_CYSCREEN);
    // Debug override: ARC_W / ARC_H force the window to a fixed size (a
    // borderless window in the top-left, not a display-mode change) so a
    // given resolution can be reproduced without touching desktop settings.
    // e.g. set ARC_W=1920 & ARC_H=1080 to repro the 1080p layout on any panel.
    if (const char *ow = debug_override("ARC_W")) {
      int v = atoi(ow);
      if (v > 0) m_width = v;
    }
    if (const char *oh = debug_override("ARC_H")) {
      int v = atoi(oh);
      if (v > 0) m_height = v;
    }

    WNDCLASSW wc = {};
    wc.lpfnWndProc = static_window_proc;
    wc.hInstance = h_instance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hIcon = LoadIconW(h_instance, L"IDI_ICON1");
    wc.lpszClassName = L"ARC_SOLVER";
    RegisterClassW(&wc);

    m_hwnd = CreateWindowExW(WS_EX_TOPMOST, wc.lpszClassName, L"ARC",
                             WS_POPUP | WS_VISIBLE, 0, 0, m_width, m_height,
                             NULL, NULL, h_instance, NULL);

    // Start minimized when the agent drives (--minimized or saved state): never
    // show the window full screen first, so there is no flash. The
    // restore-on-activate path in handle_message brings it back when the human
    // clicks it.
    if (n_show_cmd == SW_MINIMIZE) {
      m_app_state.gui_minimized = true;
      ShowWindow(m_hwnd, SW_MINIMIZE);
      UpdateWindow(m_hwnd);
    } else {
      enter_fullscreen(m_hwnd, m_width, m_height);
      ShowWindow(m_hwnd, SW_SHOW);
      UpdateWindow(m_hwnd);
      force_focus();
    }

    DWORD delay = 0, speed = 0;
    SystemParametersInfo(SPI_GETKEYBOARDDELAY, 0, &delay, 0);
    SystemParametersInfo(SPI_GETKEYBOARDSPEED, 0, &speed, 0);
    // A STEPPER IS NOT A TYPEWRITER. These two numbers are the machine's typing
    // settings, meant for holding a letter down in a document, where waiting
    // out a long pause and then getting hundreds a second is exactly right. A
    // list of five wants the opposite: move as soon as it is clear you are
    // holding, then slowly enough to stop where you meant to. So the wait is a
    // half of the typing pause, and the rate is floored at something a hand can
    // steer whatever the machine is set to.
    m_repeat_delay_s = 0.125 * (1 + delay);
    m_repeat_interval_s = max(0.09, 0.5 / (2.5 + speed * 0.8867));
  }

  inline bool process_messages() {
    MSG msg = {};
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT)
        return false;
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
    return true;
  }

  inline void force_focus() {
    // Method 1: Standard
    SetForegroundWindow(m_hwnd);
    SetFocus(m_hwnd);

    // Method 2: AttachThreadInput Hack
    DWORD current_thread = GetCurrentThreadId();
    DWORD active_window_thread =
        GetWindowThreadProcessId(GetForegroundWindow(), NULL);
    if (current_thread != active_window_thread) {
      AttachThreadInput(active_window_thread, current_thread, TRUE);
      SetForegroundWindow(m_hwnd);
      SetFocus(m_hwnd);
      AttachThreadInput(active_window_thread, current_thread, FALSE);
    }

    // Method 3: Topmost Toggle
    SetWindowPos(m_hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
    SetWindowPos(m_hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
    SetFocus(m_hwnd);
  }

  // Single entry for actions arriving from outside the window (command port),
  // so remote commands and keyboard/mouse share one behavior.
  inline void perform(ui_action action) { execute_action(action); }

  inline HWND get_hwnd() const { return m_hwnd; }
  inline int get_width() const { return m_width; }
  inline int get_height() const { return m_height; }

  inline void tick_repeat() {
    if (m_held_action == ui_action::none) return;
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    double elapsed = (double)(now.QuadPart - m_held_start.QuadPart) / freq.QuadPart;
    double interval = m_repeat_interval_s;
    double threshold = m_held_repeating ? interval : m_repeat_delay_s;
    if (elapsed >= threshold) {
      int fires = m_held_repeating ? max(1, (int)(elapsed / interval)) : 1;
      for (int i = 0; i < fires; ++i) fire_held_action();
      m_held_start = now;
      m_held_repeating = true;
    }
  }

private:
  static LRESULT CALLBACK static_window_proc(HWND hwnd, UINT u_msg,
                                             WPARAM w_param, LPARAM l_param) {
    if (g_window_manager_ptr)
      return g_window_manager_ptr->handle_message(hwnd, u_msg, w_param, l_param);
    return DefWindowProc(hwnd, u_msg, w_param, l_param);
  }

  inline void execute_action(ui_action action) {
    switch (action) {
      case ui_action::toggle_pause:   m_solver_threading.toggle_pause(); break;
      // The drawer opens by dragging its handle, which nothing but a mouse can
      // do -- so the port could not see the one panel that lists the ops. It
      // opens all the way: pulled past the end, which the draw stops at full.
      case ui_action::toggle_op_list:
        m_app_state.op_drawer_h.store(numeric_limits<int>::max());
        m_app_state.show_op_list.store(!m_app_state.show_op_list.load());
        break;
      case ui_action::toggle_programs:
        m_app_state.show_whole_programs.store(!m_app_state.show_whole_programs.load());
        break;
      case ui_action::op_tab_get:       m_app_state.op_tab = (int)op_role::get; break;
      case ui_action::op_tab_reduce:    m_app_state.op_tab = (int)op_role::reduce; break;
      case ui_action::op_tab_select:    m_app_state.op_tab = (int)op_role::select; break;
      case ui_action::op_tab_translate: m_app_state.op_tab = (int)op_role::translate; break;
      case ui_action::op_tab_map:       m_app_state.op_tab = (int)op_role::map; break;
      case ui_action::op_tab_generate:  m_app_state.op_tab = (int)op_role::generate; break;
      case ui_action::op_tab_combine:   m_app_state.op_tab = (int)op_role::combine; break;
      case ui_action::toggle_log: {
        logging_mode m = m_app_state.log_mode;
        m_app_state.log_mode = (m == logging_mode::none) ? logging_mode::stream : logging_mode::none;
        break;
      }
      case ui_action::prev_puzzle:    m_solver_threading.change_puzzle(-1); break;
      case ui_action::next_puzzle:    m_solver_threading.change_puzzle(1); break;
      case ui_action::first_puzzle: {
        // Jump to the first puzzle. change_puzzle takes a relative delta, so
        // step back by the current index to land exactly on 0.
        int cur = m_solver_threading.current_puzzle_idx();
        if (cur > 0) m_solver_threading.change_puzzle(-cur);
        break;
      }
      case ui_action::last_puzzle: {
        int cur = m_solver_threading.current_puzzle_idx();
        int last = m_solver_threading.get_puzzle_count() - 1;
        if (last >= 0 && cur != last) m_solver_threading.change_puzzle(last - cur);
        break;
      }
      case ui_action::edit_id: {
        // Toggle the inline ID editor. Opening seeds it empty; clicking again
        // (or Esc) closes it without jumping.
        std::lock_guard<std::mutex> lk(m_app_state.id_edit_mtx);
        auto &ed = m_app_state.id_edit;
        ed.active = !ed.active;
        ed.error = false;
        ed.buffer.clear();
        break;
      }
      case ui_action::copy_id:
        set_clipboard_text(m_solver_threading.current_puzzle_id());
        break;
      case ui_action::paste_id: {
        // Paste an id from the clipboard and jump straight to it (no edit mode).
        string clip = get_clipboard_text();
        string id;
        for (char c : clip)
          if (isalnum((unsigned char)c) || c == '_' || c == '-')
            id.push_back(c);
        m_solver_threading.jump_to_puzzle_id(id);
        break;
      }
      case ui_action::toggle_mute: {
        auto &sm = sound_manager::get();
        sm.set_muted(!sm.is_muted());
        break;
      }
      case ui_action::import_sol:     m_solver_threading.import_solution(); break;
      case ui_action::export_sol:     m_solver_threading.export_solution(); break;
      // The run itself follows this switch (35_benchmark.hpp).
      case ui_action::toggle_benchmark:
        m_app_state.is_benchmarking = !m_app_state.is_benchmarking;
        break;
      case ui_action::inc_limit:
        m_app_state.benchmark_puzzle_limit = max(1, min(m_solver_threading.get_puzzle_count(), m_app_state.benchmark_puzzle_limit + 1));
        break;
      case ui_action::dec_limit:
        m_app_state.benchmark_puzzle_limit = max(1, m_app_state.benchmark_puzzle_limit - 1);
        break;
      case ui_action::reset_solver:   m_solver_threading.reset_solver(); break;
      // Dataset checkmarks: flip one set's or filter's bit and apply.
      case ui_action::toggle_ds_train:
        m_solver_threading.set_datasets(m_solver_threading.get_dataset_flags() ^ DS_TRAIN);
        break;
      case ui_action::toggle_ds_eval:
        m_solver_threading.set_datasets(m_solver_threading.get_dataset_flags() ^ DS_EVAL);
        break;
      // One view at a time; the one on screen turns the overlay off.
      case ui_action::toggle_inference_facts:
      case ui_action::toggle_inference_lines: {
        const int view = action == ui_action::toggle_inference_facts ? global_app_state::INFERENCE_FACTS
                                                                     : global_app_state::INFERENCE_LINES;
        m_app_state.inference_view.store(m_app_state.inference_view.load() == view
                                             ? global_app_state::INFERENCE_OFF : view);
        break;
      }
      // Every stepper carries its own count, told to it by the draw — the only
      // place that knows how long this puzzle's list is — so a step here is
      // just a step and each one stops against its own end.
      case ui_action::cycle_coarseness:      m_app_state.inference_rank.step(1); break;
      case ui_action::cycle_coarseness_back: m_app_state.inference_rank.step(-1); break;
      case ui_action::step_edge:         m_app_state.inference_edge.step(1); break;
      case ui_action::step_edge_back:    m_app_state.inference_edge.step(-1); break;
      case ui_action::step_piece:        m_app_state.inference_piece.step(1); break;
      case ui_action::step_piece_back:   m_app_state.inference_piece.step(-1); break;
      case ui_action::step_motif:        m_app_state.inference_motif.step(1); break;
      case ui_action::step_motif_back:   m_app_state.inference_motif.step(-1); break;
      case ui_action::step_dropped:      m_app_state.inference_dropped.step(1); break;
      case ui_action::step_dropped_back: m_app_state.inference_dropped.step(-1); break;
      case ui_action::step_left_over:      m_app_state.inference_left_over.step(1); break;
      case ui_action::step_left_over_back: m_app_state.inference_left_over.step(-1); break;
      case ui_action::toggle_ds_test:
        m_solver_threading.set_datasets(m_solver_threading.get_dataset_flags() ^ DS_TEST);
        break;
      case ui_action::toggle_ds_custom:
        m_solver_threading.set_datasets(m_solver_threading.get_dataset_flags() ^ DS_CUSTOM);
        break;
      case ui_action::toggle_ds_agi_1:
        m_solver_threading.set_datasets(m_solver_threading.get_dataset_flags() ^ DS_HIDE_AGI_1);
        break;
      case ui_action::toggle_ds_agi_2:
        m_solver_threading.set_datasets(m_solver_threading.get_dataset_flags() ^ DS_HIDE_AGI_2);
        break;
      case ui_action::exit_app:       PostQuitMessage(0); break;
      case ui_action::minimize_app:   ShowWindow(m_hwnd, SW_MINIMIZE); break;
      default: break;
    }
  }

  inline LRESULT handle_message(HWND hwnd, UINT u_msg, WPARAM w_param,
                                LPARAM l_param) {
    // WM_SIZE is the truthful minimize signal: SIZE_MINIMIZED on the way down,
    // SIZE_RESTORED on the way back up. Track it so the flag always says whether
    // the window is minimized, for both the restore path and settings.
    if (u_msg == WM_SIZE) {
      if (w_param == SIZE_MINIMIZED) {
        m_app_state.gui_minimized = true;
      } else if (w_param == SIZE_RESTORED && m_app_state.gui_minimized) {
        m_app_state.gui_minimized = false;
        int w = GetSystemMetrics(SM_CXSCREEN);
        int h = GetSystemMetrics(SM_CYSCREEN);
        enter_fullscreen(hwnd, w, h);
        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, w, h, SWP_SHOWWINDOW);
        force_focus();
        m_app_state.needs_resize = true;
      }
      return 0;
    }
    if (u_msg == WM_DESTROY) {
      PostQuitMessage(0);
      return 0;
    }
    if (u_msg == WM_KEYDOWN) {
      bool first_press = !(l_param & (1 << 30));
      bool ctrl = GetKeyState(VK_CONTROL) & 0x8000;

      // While the ID editor is open, the keyboard drives the text field, not
      // the normal hotkeys.
      bool editing;
      {
        std::lock_guard<std::mutex> lk(m_app_state.id_edit_mtx);
        editing = m_app_state.id_edit.active;
      }

      // Copy and paste answer from anywhere, in any mode. Copy takes the live
      // ID (the in-progress buffer is rarely what is wanted); paste fills the
      // buffer while editing, and jumps straight to the id when not.
      if (ctrl && w_param == 'C') {
        execute_action(ui_action::copy_id);
        return 0;
      }
      if (ctrl && w_param == 'V') {
        if (!editing) {
          execute_action(ui_action::paste_id);
          return 0;
        }
        string clip = get_clipboard_text();
        std::lock_guard<std::mutex> lk(m_app_state.id_edit_mtx);
        for (char c : clip)
          if (isalnum((unsigned char)c) || c == '_' || c == '-')
            m_app_state.id_edit.buffer.push_back(c);
        m_app_state.id_edit.error = false;
        return 0;
      }
      if (editing) {
        if (w_param == VK_RETURN) {
          commit_id_edit();
        } else if (w_param == VK_ESCAPE) {
          std::lock_guard<std::mutex> lk(m_app_state.id_edit_mtx);
          m_app_state.id_edit.active = false;
          m_app_state.id_edit.error = false;
          m_app_state.id_edit.buffer.clear();
        } else if (w_param == VK_BACK) {
          std::lock_guard<std::mutex> lk(m_app_state.id_edit_mtx);
          if (!m_app_state.id_edit.buffer.empty())
            m_app_state.id_edit.buffer.pop_back();
          m_app_state.id_edit.error = false;
        }
        return 0; // swallow: no hotkeys, no OS beep handling here
      }

      ui_action action = key_to_action(w_param, (GetKeyState(VK_SHIFT) & 0x8000) != 0);

      if (action != ui_action::none) {
        m_app_state.active_ui_action = action;
        if (first_press) {
          execute_action(action);
          // Repeatable actions hold-to-repeat through the same timed
          // mechanism the mouse uses, so the keyboard acts like the mouse on
          // hold. We ignore the OS key-repeat (the 1<<30 repeat bit) entirely
          // and drive repeats from tick_repeat() instead, giving consistent
          // ramp-up regardless of system keyboard settings.
          if (is_repeatable(action)) {
            m_held_action = action;
            m_held_key = w_param;
            QueryPerformanceCounter(&m_held_start);
            m_held_repeating = false;
          }
        }
      }
    }
    if (u_msg == WM_KEYUP) {
      ui_action prev = m_app_state.active_ui_action.exchange(ui_action::none);
      // The repeat belongs to the KEY that started it, not to the action it
      // maps to: letting go of shift before the letter would otherwise change
      // the action mid-hold and leave the repeat running forever.
      if (m_held_key && w_param == m_held_key) {
        m_held_action = ui_action::none;
        m_held_key = 0;
      }
      if (prev == ui_action::exit_app) PostQuitMessage(0);
    }
    // A HOLD ENDS WHEN THE WINDOW STOPS LISTENING. The key-up that would end it
    // goes to whoever took the focus, so without this the stepper keeps running
    // on its own after an alt-tab, with nothing on screen able to stop it.
    if (u_msg == WM_KILLFOCUS) {
      m_held_action = ui_action::none;
      m_held_key = 0;
    }
    if (u_msg == WM_CHAR) {
      // Printable characters for the ID editor. Control chars (Enter, Esc,
      // Backspace) and Ctrl shortcuts are handled in WM_KEYDOWN; here we only
      // accept ID-valid characters. ARC ids are alnum; allow -/_ defensively.
      std::lock_guard<std::mutex> lk(m_app_state.id_edit_mtx);
      if (m_app_state.id_edit.active) {
        char c = (char)w_param;
        if (isalnum((unsigned char)c) || c == '_' || c == '-') {
          if (m_app_state.id_edit.buffer.size() < 32)
            m_app_state.id_edit.buffer.push_back(c);
          m_app_state.id_edit.error = false;
        }
        return 0;
      }
    }
    if (u_msg == WM_MOUSEMOVE) {
      int x = (short)LOWORD(l_param);
      int y = (short)HIWORD(l_param);
      // A held track follows the cursor, so one drag sweeps every stop. It keeps
      // following past its own ends, which just pin to the first and last stop.
      if (m_scrub) {
        scrub_to(x);
        return 0;
      }
      // Handle drag: the drawer's bottom edge follows the cursor. Dragging down
      // opens/expands; dragging up shrinks; releasing near the top closes. Any
      // move makes it a drag (consumed), so the ensuing button-up toggle doesn't
      // fire.
      if (m_dragging_handle) {
        m_drag_consumed = true;
        m_app_state.show_op_list.store(true);
        m_app_state.active_ui_action.store(ui_action::none); // cancel pending toggle
        int hmax = m_app_state.op_drawer_h_max.load();
        int nh = y + m_drag_offset;
        if (nh < 0) nh = 0;
        if (hmax > 0 && nh > hmax) nh = hmax;
        m_app_state.op_drawer_h.store(nh);
      }
      // Handle "hot" when the cursor is near it (within ~2 handle-heights) so it
      // fades in from dim; otherwise dim. Skip while dragging (always hot).
      {
        int hx = m_app_state.op_handle_x.load();
        int hy = m_app_state.op_handle_y.load();
        int hw = m_app_state.op_handle_w.load();
        int hh = m_app_state.op_handle_h.load();
        bool is_near = hw > 0 && x >= hx && x <= hx + hw &&
                       y >= hy && y <= hy + hh;
        m_app_state.op_handle_hot.store(is_near || m_dragging_handle);
      }
      uint64_t hit = 0;
      {
        std::lock_guard<std::mutex> lk(m_app_state.node_zones_mtx);
        for (const auto &z : m_app_state.node_zones) {
          if (x >= z.x && x <= z.x + z.w && y >= z.y && y <= z.y + z.h) {
            hit = z.node_hash;
            break;
          }
        }
      }
      m_app_state.hovered_hash.store(hit);
      return 0;
    }
    // THE WHEEL SCROLLS THE REGION UNDER THE CURSOR, the one drawn last where
    // two overlap, by as many of its lines a notch as Windows is set to scroll,
    // or by its whole height where Windows is set to scroll pages.
    if (u_msg == WM_MOUSEWHEEL) {
      POINT at{(short)LOWORD(l_param), (short)HIWORD(l_param)};
      ScreenToClient(hwnd, &at);
      UINT lines = 0;
      SystemParametersInfo(SPI_GETWHEELSCROLLLINES, 0, &lines, 0);
      const auto &zones = m_app_state.scroll_zones;
      for (auto z = zones.rbegin(); z != zones.rend(); ++z) {
        if (at.x < z->x || at.x >= z->x + z->w || at.y < z->y || at.y >= z->y + z->h) continue;
        const int step = lines == WHEEL_PAGESCROLL ? z->h : (int)lines * z->line;
        z->which->by(-GET_WHEEL_DELTA_WPARAM(w_param) * step / WHEEL_DELTA);
        break;
      }
      return 0;
    }
    if (u_msg == WM_LBUTTONDOWN) {
      int x = (short)LOWORD(l_param);
      int y = (short)HIWORD(l_param);
      // Begin a potential drag on the handle (open or closed). A move past the
      // deadzone turns it into a resize/open drag; a release without moving is a
      // plain toggle click.
      m_dragging_handle = false;
      m_drag_consumed = false;
      // A press on a track grabs it and lands on the stop under the cursor;
      // the drawer, when it is open over one, swallows the press as it does for
      // buttons. Grabbing takes the whole gesture, so nothing else sees it.
      if (!drawer_blocks_zone(y, ui_action::none)) {
        for (const auto &z : m_app_state.slider_zones) {
          if (x >= z.x && x <= z.x + z.w && y >= z.y && y <= z.y + z.h) {
            m_scrub = z.which;
            m_scrub_x = z.x;
            m_scrub_w = z.w;
            m_scrub_stops = z.stops;
            m_scrub_knob = z.knob;
            scrub_to(x);
            return 0;
          }
        }
      }
      // Drag from the handle, or from any empty space inside the open panel
      // (not over a tab/op zone).
      bool on_empty_panel = m_app_state.show_op_list.load() &&
                            y >= 0 && y < m_app_state.op_drawer_bottom.load() &&
                            !point_over_click_zone(x, y);
      if (point_in_handle(x, y) || on_empty_panel) {
        m_dragging_handle = true;
        // Move the bottom edge WITH the cursor (preserve grab offset), not TO it.
        int cur_bottom = m_app_state.show_op_list.load()
                             ? m_app_state.op_drawer_bottom.load() : 0;
        m_drag_offset = cur_bottom - y;
      }
      for (const auto &zone : m_app_state.click_zones) {
        if (drawer_blocks_zone(y, zone.action)) continue;
        if (x >= zone.x && x <= zone.x + zone.w && y >= zone.y && y <= zone.y + zone.h) {
          m_app_state.active_ui_action = zone.action;
          if (is_repeatable(zone.action)) {
            execute_action(zone.action);
            m_held_action = zone.action;
            m_held_key = 0;   // a mouse hold is owned by the button, not a key
            QueryPerformanceCounter(&m_held_start);
            m_held_repeating = false;
          }
          break;
        }
      }
      return 0;
    }
    if (u_msg == WM_LBUTTONUP) {
      m_held_action = ui_action::none;
      m_held_key = 0;
      // Letting go of a track ends the gesture there; the press never reached
      // the button and drag handling below, so neither may act on the release.
      if (m_scrub) {
        m_scrub = nullptr;
        return 0;
      }
      bool was_drag = m_dragging_handle;
      bool drag_consumed = m_drag_consumed;
      m_dragging_handle = false;
      m_drag_consumed = false;
      int x = (short)LOWORD(l_param);
      int y = (short)HIWORD(l_param);
      ui_action current_action = m_app_state.active_ui_action.exchange(ui_action::none);
      // A completed drag already sized the drawer; swallow the click so it
      // doesn't toggle. Snap on release: past the halfway point -> fully open;
      // otherwise -> closed.
      if (was_drag && drag_consumed) {
        int pmax = m_app_state.op_drawer_h_max.load();
        int cur = m_app_state.op_drawer_h.load();
        if (pmax > 0 && cur >= pmax / 2) {
          m_app_state.op_drawer_h.store(pmax);   // snap open
          m_app_state.show_op_list.store(true);
        } else {
          m_app_state.op_drawer_h.store(0);      // snap closed
          m_app_state.show_op_list.store(false);
        }
        return 0;
      }

      if (current_action != ui_action::none && !is_repeatable(current_action)) {
        for (const auto &zone : m_app_state.click_zones) {
          if (drawer_blocks_zone(y, zone.action)) continue;
          if (zone.action == current_action && x >= zone.x && x <= zone.x + zone.w && y >= zone.y && y <= zone.y + zone.h) {
            execute_action(zone.action);
            break;
          }
        }
      }
      return 0;
    }
    return DefWindowProc(hwnd, u_msg, w_param, l_param);
  }

  global_app_state &m_app_state;
  solver_session &m_solver_threading;

  HWND m_hwnd = NULL;
  int m_width = 0;
  int m_height = 0;
  ui_action m_held_action = ui_action::none;
  WPARAM m_held_key = 0;   // key that owns the repeat; 0 when the mouse owns it
  bool m_held_repeating = false;

  // The track being scrubbed, null when none is. Its geometry is copied on the
  // press: the zone list is rebuilt every frame, so the zone itself won't keep.
  track* m_scrub = nullptr;
  int m_scrub_x = 0, m_scrub_w = 0, m_scrub_stops = 0, m_scrub_knob = 0;

  // Put the held track on the stop under x, pinned to its ends, then store what
  // that stop means. The handle is held by its middle and travels the track less
  // its own width, so this is the exact inverse of where the handle draws.
  inline void scrub_to(int x) {
    if (!m_scrub || m_scrub_stops <= 0) return;
    int travel = m_scrub_w - m_scrub_knob;
    int stop = 0;
    if (travel > 0 && m_scrub_stops > 1) {
      int rel = x - m_scrub_x - m_scrub_knob / 2;
      stop = (rel * (m_scrub_stops - 1) + travel / 2) / travel;
      if (stop < 0) stop = 0;
      if (stop >= m_scrub_stops) stop = m_scrub_stops - 1;
    }
    m_scrub->at.store(track::from_stop(stop, m_scrub_stops - 2));
  }
  LARGE_INTEGER m_held_start{};

  // Drag-to-open state for the op-palette handle. Set on LBUTTONDOWN over the
  // handle; any move opens the drawer and consumes the gesture (so the
  // button-up toggle doesn't immediately re-close it).
  bool m_dragging_handle = false;
  bool m_drag_consumed = false;
  int  m_drag_offset = 0; // drawer_bottom - grab_y, so the edge moves with, not to, the cursor

  inline bool point_in_handle(int x, int y) const {
    int hx = m_app_state.op_handle_x.load();
    int hy = m_app_state.op_handle_y.load();
    int hw = m_app_state.op_handle_w.load();
    int hh = m_app_state.op_handle_h.load();
    return hw > 0 && x >= hx && x <= hx + hw && y >= hy && y <= hy + hh;
  }

  // True if (x,y) is over any registered click zone (a tab/op box), so a
  // panel-body drag can tell empty space from a control.
  inline bool point_over_click_zone(int x, int y) const {
    for (const auto &z : m_app_state.click_zones)
      if (x >= z.x && x <= z.x + z.w && y >= z.y && y <= z.y + z.h)
        return true;
    return false;
  }
  double m_repeat_delay_s = 0;
  double m_repeat_interval_s = 0;

  inline void set_clipboard_text(const string &text) {
    if (text.empty() || !OpenClipboard(m_hwnd)) return;
    EmptyClipboard();
    HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, text.size() + 1);
    if (h) {
      // The allocation succeeding does not mean the lock will. Nobody owns the
      // block until SetClipboardData below, so give it back rather than write
      // to nothing.
      char *dst = (char *)GlobalLock(h);
      if (!dst) { GlobalFree(h); CloseClipboard(); return; }
      memcpy(dst, text.c_str(), text.size() + 1);
      GlobalUnlock(h);
      SetClipboardData(CF_TEXT, h); // clipboard now owns h
    }
    CloseClipboard();
  }

  inline string get_clipboard_text() {
    string out;
    if (!OpenClipboard(m_hwnd)) return out;
    HANDLE h = GetClipboardData(CF_TEXT);
    if (h) {
      const char *src = (const char *)GlobalLock(h);
      if (src) { out = src; GlobalUnlock(h); }
    }
    CloseClipboard();
    return out;
  }

  // Commit the ID editor: try to jump to the typed ID. On success the editor
  // closes; on no match it stays open and flashes (error = true).
  inline void commit_id_edit() {
    string id;
    {
      std::lock_guard<std::mutex> lk(m_app_state.id_edit_mtx);
      id = m_app_state.id_edit.buffer;
    }
    bool ok = m_solver_threading.jump_to_puzzle_id(id);
    std::lock_guard<std::mutex> lk(m_app_state.id_edit_mtx);
    if (ok) {
      m_app_state.id_edit.active = false;
      m_app_state.id_edit.error = false;
      m_app_state.id_edit.buffer.clear();
    } else {
      m_app_state.id_edit.error = true;
    }
  }

  // Shift is the "other way" modifier: on a key that steps a counter it picks
  // the back direction, on I the other view, everywhere else it changes nothing.
  static inline ui_action key_to_action(WPARAM w_param, bool shift) {
    switch (w_param) {
      case VK_ESCAPE: return ui_action::exit_app;
      case VK_LEFT:   return ui_action::prev_puzzle;
      case VK_RIGHT:  return ui_action::next_puzzle;
      case VK_SPACE:  return ui_action::toggle_pause;
      case 'L':       return ui_action::toggle_log;
      case 'R':       return ui_action::reset_solver;
      // Import moved off 'I' when the inference overlay took it.
      case 'N':       return ui_action::import_sol;
      case 'E':       return ui_action::export_sol;
      case 'B':       return ui_action::toggle_benchmark;
      // The overlay: show its facts or its lines, step the coarseness, flip the
      // direction, walk the lines one at a time. Hold to keep stepping, shift to
      // step back.
      case 'I':       return shift ? ui_action::toggle_inference_lines : ui_action::toggle_inference_facts;
      case 'C':       return shift ? ui_action::cycle_coarseness_back : ui_action::cycle_coarseness;
      case 'S':       return shift ? ui_action::step_edge_back : ui_action::step_edge;
      case 'P':       return shift ? ui_action::step_piece_back : ui_action::step_piece;
      case 'X':       return shift ? ui_action::step_dropped_back : ui_action::step_dropped;
      case 'O':       return shift ? ui_action::step_motif_back : ui_action::step_motif;
      case 'U':       return shift ? ui_action::step_left_over_back : ui_action::step_left_over;
      case 'M':       return ui_action::toggle_mute;
      default:        return ui_action::none;
    }
  }

  static inline bool is_repeatable(ui_action a) {
    return a == ui_action::prev_puzzle || a == ui_action::next_puzzle ||
           a == ui_action::inc_limit   || a == ui_action::dec_limit   ||
           a == ui_action::cycle_coarseness || a == ui_action::cycle_coarseness_back ||
           a == ui_action::step_edge        || a == ui_action::step_edge_back ||
           a == ui_action::step_piece       || a == ui_action::step_piece_back ||
           a == ui_action::step_dropped     || a == ui_action::step_dropped_back ||
           a == ui_action::step_motif       || a == ui_action::step_motif_back ||
           a == ui_action::step_left_over   || a == ui_action::step_left_over_back;
  }

  static inline bool is_op_tab_action(ui_action a) {
    return a == ui_action::op_tab_get || a == ui_action::op_tab_reduce ||
           a == ui_action::op_tab_select || a == ui_action::op_tab_translate ||
           a == ui_action::op_tab_map || a == ui_action::op_tab_generate ||
           a == ui_action::op_tab_combine;
  }

  // When the op-palette drawer is open, it overlays the header band; clicks that
  // land within the drawer (y above its bottom edge) must NOT fall through to
  // the header buttons underneath (EXIT, etc.). Only the drawer's own tab zones
  // are allowed there.
  inline bool drawer_blocks_zone(int y, ui_action action) {
    if (!m_app_state.show_op_list.load()) return false;
    if (is_op_tab_action(action)) return false;
    return y < m_app_state.op_drawer_bottom.load();
  }

  inline void fire_held_action() {
    execute_action(m_held_action);
  }
};

}
