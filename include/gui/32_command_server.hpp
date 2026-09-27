#pragma once
// Local control port: lets a script (or Claude) drive the GUI over TCP.
// Protocol: connect to 127.0.0.1:<port>, send one command line, receive one
// reply line ("ok ..." or "error ..."), connection closes. The first app takes
// 2026 and each later one the next free port up. Pure transport:
// what commands mean is decided by the handler wired up in main().
// Any include order works: WIN32_LEAN_AND_MEAN (CMakeLists.txt) keeps
// windows.h from bringing in the old winsock.h that winsock2.h clashes with.
#include <winsock2.h>
// winsock2.h pulls windows.h, whose GDI macro TRANSPARENT would shadow the
// arc::TRANSPARENT constant in 1_readings.hpp. Nothing here uses the macro.
#undef TRANSPARENT
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include "11_text_format.hpp"

namespace arc {
using namespace std;

class command_server {
public:
  // Takes the command line, returns the reply line.
  using handler = function<string(const string &)>;

  // Takes the first port to try. A second app walks upward to the next free one,
  // so two instances never fight over 2026.
  inline bool init(unsigned short first_port, handler h) {
    m_handler = move(h);
    filesystem::create_directories("logs");

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
      open_log(first_port);
      return fail("WSAStartup failed");
    }

    for (int port = first_port; port < first_port + port_scan && port < 65536; ++port) {
      m_listen = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (m_listen == INVALID_SOCKET) {
        open_log(first_port);
        return fail("socket() failed");
      }

      // Local-only: bind to the loopback address so nothing off-machine can connect.
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons((unsigned short)port);
      addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      // ::bind is the socket call; plain bind would find std::bind here.
      // A taken port is the expected case with a second app, not an error.
      if (::bind(m_listen, (sockaddr *)&addr, (int)sizeof(addr)) == SOCKET_ERROR ||
          listen(m_listen, SOMAXCONN) == SOCKET_ERROR) {
        closesocket(m_listen);
        m_listen = INVALID_SOCKET;
        continue;
      }

      m_port = (unsigned short)port;
      // Non-blocking accept: the frame-loop poll costs nothing when nobody connects.
      u_long nonblock = 1;
      ioctlsocket(m_listen, FIONBIO, &nonblock);
      open_log(m_port);
      log_line(to_text("listening on 127.0.0.1:%d", m_port));
      return true;
    }

    open_log(first_port);
    return fail(to_text("no free port in %d..%d", first_port,
                        first_port + port_scan - 1));
  }

  // Which port this app actually got, so callers can report it.
  inline unsigned short port() const { return m_port; }

  // Put a line of our own in the receipt log, next to the commands.
  inline void note(const string &msg) { log_line(msg); }

  // Called once per frame on the main thread; serves at most one command.
  inline void poll() {
    if (m_listen == INVALID_SOCKET) return;
    SOCKET client = accept(m_listen, nullptr, nullptr);
    if (client == INVALID_SOCKET) return;

    // The client is a one-shot scripted sender, so a short blocking read is
    // fine: worst case one slow frame, and only when someone connected.
    u_long block = 0;
    ioctlsocket(client, FIONBIO, &block);
    DWORD timeout_ms = 500;
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout_ms, sizeof(timeout_ms));

    string cmd = read_line(client);
    string reply = cmd.empty() ? "error empty command" : m_handler(cmd);
    reply.push_back('\n');
    send(client, reply.c_str(), (int)reply.size(), 0);
    closesocket(client);

    reply.pop_back();
    log_line(to_text("%s -> %s", cmd.empty() ? "(empty)" : cmd.c_str(), reply.c_str()));
  }

  inline void shutdown() {
    if (m_listen != INVALID_SOCKET) {
      closesocket(m_listen);
      m_listen = INVALID_SOCKET;
      WSACleanup();
      log_line("stopped");
    }
  }

private:
  // How many ports to try before giving up, counting from the one asked for.
  static constexpr int port_scan = 10;

  SOCKET m_listen = INVALID_SOCKET;
  unsigned short m_port = 0;
  handler m_handler;
  ofstream m_log;

  // One receipt file per port: a second app must not overwrite the first's
  // record. Fresh each launch, so the log is this session, not history.
  inline void open_log(unsigned short port) {
    if (m_log.is_open()) return;
    m_log.open(to_text("logs/commands_%d.txt", port), ios::trunc);
  }

  // Read up to the first newline (or timeout/limit); trims trailing \r\n.
  inline string read_line(SOCKET s) {
    string line;
    char c;
    while (line.size() < 256 && recv(s, &c, 1, 0) == 1 && c != '\n')
      line.push_back(c);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
      line.pop_back();
    return line;
  }

  // App-clock receipt log: the trustworthy record of what was seen and done.
  inline void log_line(const string &msg) {
    if (!m_log.is_open()) return;
    time_t now_t = chrono::system_clock::to_time_t(chrono::system_clock::now());
    tm local_tm{};
    localtime_s(&local_tm, &now_t);
    char ts[24];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &local_tm);
    m_log << ts << " | " << msg << '\n';
    m_log.flush();
  }

  inline bool fail(const string &why) {
    log_line("error: " + why);
    if (m_listen != INVALID_SOCKET) { closesocket(m_listen); m_listen = INVALID_SOCKET; }
    WSACleanup();
    return false;
  }
};

}
