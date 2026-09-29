@echo off
setlocal
cd /d "%~dp0\..\.."

python perception_cli.py --map forest-of-dead-trees-2 monster-review ^
    --images-only ^
    --split all ^
    --queue all ^
    --frame-id-prefix pilot-

if errorlevel 1 (
    echo.
    echo The Forest of Dead Trees 2 original-label review GUI exited with an error.
    pause
)

endlocal
