@echo off
echo Starting Cricket Prediction Frontend...
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
    echo.
)

echo Starting React development server...
echo Open your browser to http://localhost:3000
echo.
npm start
