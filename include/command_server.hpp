#pragma once
// Local control port: lets a script (or Claude) drive the GUI over TCP.
// Protocol: connect to 127.0.0.1:2026, send one command line, receive one
// reply line ("ok ..." or "error ..."), connection closes. Pure transport:
// what commands mean is decided by the handler wired up in main().
// winsock2.h must precede windows.h, so include this header before the
// other Win32 headers.
#include <winsock2.h>
// winsock2.h pulls windows.h, whose GDI macro TRANSPARENT would shadow the
// arc::TRANSPARENT constant in environment.hpp. Nothing here uses the macro.
#undef TRANSPARENT
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include "fmt/format.h"

namespace arc {
using namespace std;

class command_server {
public:
  // Takes the command line, returns the reply line.
  using handler = function<string(const string &)>;

  inline bool init(unsigned short port, handler h) {
    m_handler = move(h);
    filesystem::create_directories("logs");
    // Fresh receipts each launch: the log records this session, not history.
    m_log.open("logs/commands.txt", ios::trunc);

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return fail("WSAStartup failed");
    m_listen = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (m_listen == INVALID_SOCKET) return fail("socket() failed");

    // Local-only: bind to the loopback address so nothing off-machine can connect.
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    // ::bind is the socket call; plain bind would find std::bind here.
    if (::bind(m_listen, (sockaddr *)&addr, (int)sizeof(addr)) == SOCKET_ERROR)
      return fail(fmt::format("bind to port {} failed (in use?)", port));
    if (listen(m_listen, SOMAXCONN) == SOCKET_ERROR) return fail("listen() failed");

    // Non-blocking accept: the frame-loop poll costs nothing when nobody connects.
    u_long nonblock = 1;
    ioctlsocket(m_listen, FIONBIO, &nonblock);
    log_line(fmt::format("listening on 127.0.0.1:{}", port));
    return true;
  }

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
    log_line(fmt::format("{} -> {}", cmd.empty() ? "(empty)" : cmd, reply));
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
  SOCKET m_listen = INVALID_SOCKET;
  handler m_handler;
  ofstream m_log;

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
