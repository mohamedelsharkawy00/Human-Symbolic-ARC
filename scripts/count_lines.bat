@echo off
setlocal

:: Run the PowerShell script with a one-time execution policy bypass so the system policy is not changed.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0count_lines.ps1"
exit /b %errorlevel%
