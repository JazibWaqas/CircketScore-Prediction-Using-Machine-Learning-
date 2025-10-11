@echo off
title Cricket Predictor - Share via Ngrok
color 0A

echo.
echo ========================================
echo   CRICKET PREDICTOR - ONE URL
echo ========================================
echo.

echo [1/3] Building frontend...
cd dashboard\frontend
call npm run build
cd ..\..

echo.
echo [2/3] Starting server (backend + frontend)...
start "Cricket Predictor" cmd /k "cd dashboard\backend && python app.py"
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Creating public URL...
start "Share This URL" cmd /k "echo === COPY AND SHARE THIS URL === && echo. && ngrok http 5002"

echo.
echo ========================================
echo   READY!
echo ========================================
echo.
echo Copy the URL from "Share This URL" window
echo Share it with anyone - works everywhere!
echo.
pause

