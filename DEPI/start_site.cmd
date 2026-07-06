@echo off
setlocal
title AI Medical Center Server

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Could not find .venv\Scripts\python.exe
    echo Run this command from the project folder after creating the virtual environment.
    pause
    exit /b 1
)

echo Starting AI Medical Center on http://127.0.0.1:8000/
echo Keep this window open while using the site.
echo Logs:
echo   %~dp0server.log
echo   %~dp0server.err
echo.

".venv\Scripts\python.exe" run_server.py

echo.
echo Server stopped with exit code %ERRORLEVEL%.
pause
