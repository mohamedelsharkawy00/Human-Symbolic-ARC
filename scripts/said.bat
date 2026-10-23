@echo off
setlocal
cd /d "%~dp0.."

:: EVERY PUZZLE READ, SCORED AND WRITTEN DOWN, in one run: builds arc_said
:: through scripts\build.bat and runs it, writing said.txt, tally.txt and
:: logs\inference\<id>.txt into out\x64-Release\, or out\<label>\ when named.
::   scripts\said.bat [--as <label>] [--solved [label]] [id...]
:: Then compare two runs with scripts\readouts.py (the unnamed one is x64-Release).
call "%~dp0build.bat" x64-Release NOLAUNCH arc_said
if errorlevel 1 exit /b 1
"out\x64-Release\arc_said.exe" %*
