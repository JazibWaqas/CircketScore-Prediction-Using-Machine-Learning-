@echo off
echo ========================================
echo Cricket Score Prediction System
echo ========================================
echo.

echo Starting Database...
cd database
start /B python app_minimal.py
echo Database API started on http://localhost:5000
echo.

echo Starting Frontend...
cd ..\frontend
start /B npm start
echo Frontend starting on http://localhost:3000
echo.

echo ========================================
echo System is starting up...
echo ========================================
echo.
echo Database API: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to stop all services...
pause > nul

echo Stopping services...
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul
echo All services stopped.
