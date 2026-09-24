@echo off
setlocal
cd /d "%~dp0\..\.."

python perception_cli.py --map forest-of-dead-trees-2 monster-review ^
    --video "C:\projects\input-flag-inspector\saves_m\ForestOfDeadTreeSample4_LottaLich\screen.mp4" ^
    --split all ^
    --queue all ^
    --frame-id-prefix codex-sample4- ^
    --start-id codex-sample4-random-0000

if errorlevel 1 (
    echo.
    echo The Forest of Dead Trees 2 Sample4 review GUI exited with an error.
    pause
)

endlocal
