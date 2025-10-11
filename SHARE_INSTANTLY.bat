@echo off
title Cricket Predictor - Share Instantly
color 0B

echo.
echo ================================================
echo   CRICKET PREDICTOR - INSTANT PUBLIC ACCESS
echo ================================================
echo.

echo Starting services...
echo.

echo [1/3] Starting Backend (Flask API)...
start "Backend API" cmd /k "cd dashboard\backend && echo Backend running on http://localhost:5002 && python app.py"
timeout /t 5 /nobreak >nul

echo [2/3] Starting Frontend (React)...
start "Frontend" cmd /k "cd dashboard\frontend && echo Frontend running on http://localhost:3000 && npm start"
timeout /t 15 /nobreak >nul

echo [3/3] Creating Public URL with LocalTunnel...
echo.
echo ================================================
echo   COPY AND SHARE THIS URL!
echo ================================================
echo.

lt --port 3000

echo.
echo LocalTunnel closed. Press any key to exit...
pause >nul

