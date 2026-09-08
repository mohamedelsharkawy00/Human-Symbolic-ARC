"""Say whether anyone else is working on this repo right now.

Why this exists: nothing announces a second session. A fresh chat lands in the
main folder, sees an ordinary repo, and starts working in it. The signals that
know are already there -- they just have to be asked, and asking has to happen
without anyone remembering to.

So this is called from scripts\\build.bat, at the moment a collision would
matter, rather than being a ritual run every message.

Three signals, none of which can go stale, because none of them is a marker
anyone has to release:
  worktrees   -- more than one folder open on this repo
  port 2026   -- an app answering means a session is live
  uncommitted -- work in progress that a sibling session cannot see

Usage:
    python scripts/session_check.py          # print only what is worth saying
    python scripts/session_check.py --all    # print every answer, quiet or not
"""
import argparse, os, socket, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST_PORT = 2026


def run_git(*args):
    result = subprocess.run(["git", "-C", ROOT, *args],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def other_worktrees():
    """Folders open on this repo other than the one we are standing in."""
    here = os.path.normcase(ROOT)
    folders = [line.split(" ")[0] for line in run_git("worktree", "list").splitlines()]
    return [f for f in folders if os.path.normcase(os.path.normpath(f)) != here]


def app_on_port(port):
    """The state line if an app answers, otherwise None. Refused means nobody."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1) as link:
            link.sendall(b"state\n")
            return link.recv(512).decode(errors="replace").strip() or None
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="print every answer, including the quiet ones")
    args = parser.parse_args()

    worktrees = other_worktrees()
    running = [(p, s) for p in (FIRST_PORT, FIRST_PORT + 1, FIRST_PORT + 2)
               if (s := app_on_port(p))]
    uncommitted = [l for l in run_git("status", "--porcelain").splitlines() if l]

    said = False
    if worktrees:
        print(f"[*] Another folder is open on this repo: {', '.join(worktrees)}")
        said = True
    if running:
        for port, state in running:
            print(f"[*] An app is already running on {port}: {state}")
        said = True
    if uncommitted:
        print(f"[*] {len(uncommitted)} uncommitted file(s) here, a sibling session cannot see them.")
        said = True

    if args.all and not said:
        print("[*] Nobody else: one folder, no app running, nothing uncommitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
