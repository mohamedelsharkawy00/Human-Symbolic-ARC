@echo off
REM Live-edit the notes: starts the MkDocs preview server, which watches the .md
REM files on disk and auto-rebuilds + refreshes the browser on every save.
REM Edit a file in notes\, save, and the page updates itself. No manual rebuild.
REM
REM Keep this window open while editing; close it (or Ctrl+C) to stop the server.
setlocal

set "MKDOCS=%~dp0..\notes\.mkdocs"
set "PY=%MKDOCS%\.venv\Scripts\python.exe"
set "CFG=%MKDOCS%\mkdocs.yml"
set "NO_MKDOCS_2_WARNING=1"

if not exist "%PY%" (
    echo [ERROR] venv not found at "%PY%".
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
"%PY%" -m mkdocs serve -f "%CFG%" -a 127.0.0.1:8000
