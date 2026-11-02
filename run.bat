@echo off
REM NaviGlass Desktop - Windows Quick Launcher
"C:\Users\Yash Kushwah\AppData\Local\Programs\Python\Python311\python.exe" main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo NaviGlass exited with an error.
    pause
)
