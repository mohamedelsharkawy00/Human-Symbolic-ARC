# Walk tests/background_truth.txt through the app's command port and check the
# per-side background verdicts in logs/inference/<id>.txt against the recorded
# truth. Requires the app running (out/x64-Release/ARC.exe --minimized).
import re
import socket
import time

ROOT = __file__.rsplit("scripts", 1)[0]


def cmd(line):
    s = socket.create_connection(("127.0.0.1", 2026), 5)
    s.sendall((line + "\n").encode())
    r = s.recv(4096).decode().strip()
    s.close()
    return r


def load_truth():
    rows = []
    for ln in open(ROOT + "tests/background_truth.txt"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        known_wrong = parts[0] == "WRONG"
        if known_wrong:
            parts = parts[1:]
        rows.append((parts[0], parts[1], parts[2], known_wrong))
    return rows


def side_ok(expected, verdict, per_pair_mode, per_pair):
    if expected.startswith("per-pair:"):
        return per_pair_mode and per_pair == expected[9:].split(",")
    # alternatives: "none" reads as -1 (and forbids the per-pair flag)
    alts = expected.split("|")
    if verdict == "-1":
        return "none" in alts and not per_pair_mode
    return verdict in alts


def main():
    fails = 0
    rows = load_truth()
    for pid, exp_in, exp_out, known_wrong in rows:
        cmd("puzzle " + pid)
        for _ in range(50):
            if ("id=" + pid) in cmd("state"):
                break
            time.sleep(0.1)
        time.sleep(0.2)
        txt = open(ROOT + "logs/inference/%s.txt" % pid).read()
        m = re.search(r"input=(-?\d+)\s+output=(-?\d+)(\s+MODE=PER-PAIR)?", txt)
        pp = re.search(r"across_input_output([ \-\d]+)", txt)
        vin, vout, mode = m.group(1), m.group(2), bool(m.group(3))
        per_pair = pp.group(1).split()
        ok_in = side_ok(exp_in, vin, mode, per_pair)
        # per-pair mode is an input-side fact; the output column never claims it
        ok_out = side_ok(exp_out, vout, False, [])
        ok = ok_in and ok_out
        if not ok and not known_wrong:
            fails += 1
        print("OK  " if ok else "FAIL", pid,
              "input", exp_in, "->", vin + ("(per-pair)" if mode else ""),
              " output", exp_out, "->", vout)
    print("FAILS:", fails, "of", len(rows))


if __name__ == "__main__":
    main()
