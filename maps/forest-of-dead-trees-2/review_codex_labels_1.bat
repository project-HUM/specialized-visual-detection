@echo off
setlocal
cd /d "%~dp0\..\.."

python perception_cli.py --map forest-of-dead-trees-2 monster-review ^
    --split all ^
    --queue reviewed ^
    --frame-id-prefix codex-random- ^
    --frame-id-prefix codex-lich- ^
    --start-id codex-random-0000

if errorlevel 1 (
    echo.
    echo The Forest of Dead Trees 2 Codex batch 1 review GUI exited with an error.
    pause
)

endlocal
