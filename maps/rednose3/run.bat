@echo off
setlocal
cd /d "%~dp0\..\.."

python perception_cli.py --map rednose3 monster-review ^
    --split train ^
    --queue proposal_review_required

if errorlevel 1 (
    echo.
    echo The monster review GUI exited with an error.
    pause
)

endlocal
