@echo off
title Gear Tooth Undercutting Detector - Global Server Launcher
echo =======================================================
echo Starting Gear Tooth Undercutting Detector...
echo =======================================================

cd /d "%~dp0"

echo 1. Starting Web Server on port 5000...
start /b "" .\.venv\Scripts\python.exe app.py

timeout /t 3 /nobreak >nul

echo 2. Launching Global Public Cloudflare Tunnel...
.\cloudflared.exe tunnel --url http://127.0.0.1:5000 --no-autoupdate

pause
