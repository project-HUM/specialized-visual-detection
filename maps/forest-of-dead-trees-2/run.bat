@echo off
setlocal
cd /d "%~dp0\..\.."

python perception_cli.py --map forest-of-dead-trees-2 monster-review ^
    --split pilot ^
    --queue pending

if errorlevel 1 (
    echo.
    echo The Forest of Dead Trees 2 pilot review GUI exited with an error.
    pause
)

endlocal
