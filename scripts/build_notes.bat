@echo off
REM Live-edit the notes: starts the MkDocs preview server, which watches the .md
REM files on disk and auto-rebuilds + refreshes the browser on every save.
REM Edit a file in notes\, save, and the page updates itself. No manual rebuild.
REM
REM Keep this window open while editing; close it (or Ctrl+C) to stop the server.
setlocal

set "MKDOCS=%~dp0"
set "CFG=%MKDOCS%mkdocs.yml"
set "NO_MKDOCS_2_WARNING=1"

REM Uses the machine's Python, not a venv. Two packages are all this needs;
REM pip pulls the rest. Install line is in the error below, so the fix travels
REM with the failure.
python -m mkdocs --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] mkdocs is not installed for this Python. Install it with:
    echo     python -m pip install "mkdocs==1.6.1" "mkdocs-material==9.7.6"
    pause
    exit /b 1
)

echo Starting live preview at http://127.0.0.1:8000/Home.html  (Ctrl+C to stop)

REM Wait briefly for the server to come up, then open the browser exactly ONCE.
REM (Home page is Home.html, not index.html, so open it directly to avoid a 404
REM at the bare root.) PowerShell does the wait+open in one unambiguous call,
REM backgrounded so it doesn't block the server start below.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 2; Start-Process 'http://127.0.0.1:8000/Home.html'"

REM serve from the project root so it can watch files with spaces in their paths.
REM --livereload is spelled out because mkdocs' own default for it stopped being
REM trusted: it declares the off-switch first, and newer versions of the command
REM line library it uses read the FIRST one as the default, so auto-refresh was
REM silently off. Saying it out loud does not depend on any version's default.
python -m mkdocs serve --livereload -f "%CFG%" -a 127.0.0.1:8000
