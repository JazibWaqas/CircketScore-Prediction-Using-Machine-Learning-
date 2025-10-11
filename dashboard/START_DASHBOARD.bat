@echo off
echo ================================================================================
echo STARTING ODI PROGRESSIVE DASHBOARD
echo ================================================================================
echo.
echo Starting Backend (Flask API on port 5002)...
start "ODI Backend" cmd /k "cd backend && python app.py"
timeout /t 5
echo.
echo Starting Frontend (React on port 3000)...
start "ODI Frontend" cmd /k "cd frontend && npm start"
echo.
echo ================================================================================
echo Dashboard is starting...
echo - Backend: http://localhost:5002
echo - Frontend: http://localhost:3000
echo ================================================================================
echo.
echo Both terminal windows will remain open.
echo Close them when you're done testing.
pause

