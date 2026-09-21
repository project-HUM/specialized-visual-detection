@echo off
setlocal
cd /d "%~dp0"

python perception_cli.py monster-review ^
    --video "C:\projects\map-info-extractor\InputFlagInspector_2026-08-28_01-29-59\screen.mp4" ^
    --split train ^
    --queue proposal_review_required

if errorlevel 1 (
    echo.
    echo The monster review GUI exited with an error.
    pause
)

endlocal
