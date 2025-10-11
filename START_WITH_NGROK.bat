@echo off
title Cricket Predictor - Ngrok Launcher
color 0A

echo.
echo ========================================
echo   CRICKET PREDICTOR - PUBLIC ACCESS
echo ========================================
echo.
echo Starting all services...
echo.

echo [1/4] Starting Backend API...
start "Cricket Backend" cmd /k "cd dashboard\backend && echo Backend starting on http://localhost:5002 && python app.py"
timeout /t 5 /nobreak >nul

echo [2/4] Starting Frontend Dashboard...
start "Cricket Frontend" cmd /k "cd dashboard\frontend && echo Frontend starting on http://localhost:3000 && npm start"
timeout /t 10 /nobreak >nul

echo [3/4] Creating Public URL for Backend...
start "Ngrok - Backend" cmd /k "echo COPY THE HTTPS URL BELOW && echo. && ngrok http 5002"
timeout /t 3 /nobreak >nul

echo [4/4] Creating Public URL for Frontend...
start "Ngrok - Frontend" cmd /k "echo SHARE THIS URL WITH ANYONE && echo. && ngrok http 3000"
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo   ALL SERVICES STARTED!
echo ========================================
echo.
echo NEXT STEPS:
echo.
echo 1. Look for "Ngrok - Frontend" window
echo 2. Copy the HTTPS URL (e.g., https://xyz123.ngrok.io)
echo 3. Share that URL with anyone!
echo.
echo OPTIONAL: Update Frontend API
echo   - Copy backend ngrok URL from "Ngrok - Backend" window
echo   - Update dashboard/backend/config.py CORS if needed
echo.
echo Press any key to open Ngrok Dashboard...
pause >nul
start https://dashboard.ngrok.com/

