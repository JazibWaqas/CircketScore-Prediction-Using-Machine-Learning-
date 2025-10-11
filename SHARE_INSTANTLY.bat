@echo off
title Cricket Predictor - One-Click Share
color 0A

echo.
echo ====================================================================
echo   CRICKET PREDICTOR - ONE LINK SETUP
echo ====================================================================
echo.
echo Building and starting everything automatically...
echo This will take about 30-60 seconds.
echo.

echo [1/3] Building frontend for production...
cd dashboard\frontend
call npm run build
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Frontend build failed!
    echo Make sure Node.js is installed and you've run "npm install"
    pause
    exit /b 1
)
cd ..\..

echo.
echo [2/3] Starting backend (with frontend included)...
start "Cricket Predictor" cmd /k "cd dashboard\backend && echo Server starting on port 5002... && python app.py"
timeout /t 8 /nobreak >nul

echo.
echo [3/3] Creating public URL with ngrok...
start "Public URL" cmd /k "echo === SHARE THIS URL === && echo. && ngrok http 5002"
timeout /t 3 /nobreak >nul

echo.
echo ====================================================================
echo   DONE! 
echo ====================================================================
echo.
echo Look at the "Public URL" window and copy the HTTPS link
echo Example: https://abc123.ngrok-free.app
echo.
echo Share that ONE link - everything works automatically!
echo   - No setup needed
echo   - Works on laptop, phone, anywhere
echo   - Just open and use!
echo.
echo Press any key to close this window...
pause >nul
