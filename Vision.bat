@echo off
rem ---------------------------------------------------------------
rem  Vision launcher
rem  ASCII only! Cyrillic inside a .bat breaks cmd.exe parsing:
rem  it reads the file byte by byte in the OEM codepage and tries
rem  to execute fragments of multi-byte letters as commands.
rem  All Russian text is printed by app.py instead.
rem ---------------------------------------------------------------
chcp 65001 >nul
title Vision
cd /d "%~dp0"

rem the server opens the browser itself once it is ready
set VISION_OPEN=1

rem Counters, chosen model, camera credentials and the event log live in
rem data\ - the same folder the container mounts. Without this the two ways
rem of starting would keep two separate states and the counts would drift.
set VISION_DATA=%~dp0data

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo.
  echo ERROR: .venv not found next to this file.
  echo.
  pause
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0app.py"

echo.
pause
