@echo off
setlocal
cd /d "%~dp0\..\.."

python perception_cli.py --map forest-of-dead-trees-2 monster-review ^
    --images-only ^
    --split all ^
    --queue all ^
    --frame-id-prefix codex-sample4-lichcase- ^
    --start-id codex-sample4-lichcase-0000

if errorlevel 1 (
    echo.
    echo The Forest of Dead Trees 2 Lich-diversity review GUI exited with an error.
    pause
)

endlocal
