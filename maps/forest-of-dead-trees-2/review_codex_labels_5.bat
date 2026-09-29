@echo off
setlocal
cd /d "%~dp0\..\.."

echo.
echo FOREST MANUAL LABELING - BATCH 5 ^(22 HUMAN EVIDENCE FRAMES^)
echo ------------------------------------------------------------
echo Exact HUMAN incident screenshots with Codex pre-labels.
echo Inspect every visible Zombie, Hero and Lich, including obscured sprites.
echo Includes 8 Lich-positive frames and 14 false-positive examples.
echo Signs, loot and inventory icons are not Lich labels.
echo Still images only; no temporal video context is available.
echo.
echo  Ctrl+1 = Zombie     Ctrl+2 = Hero     Ctrl+3 = Lich
echo  Click the ground/feet position to add the selected type.
echo  Drag to move. Del removes. Enter confirms and advances.
echo  Left/Right revisits frames. Q or Window X saves and quits.
echo ------------------------------------------------------------
echo.

python perception_cli.py --map forest-of-dead-trees-2 monster-review ^
    --images-only ^
    --split all ^
    --queue all ^
    --frame-id-prefix codex-human-189ec300-false-positive- ^
    --frame-id-prefix codex-human-batch5-evidence- ^
    --start-id codex-human-189ec300-false-positive-0000

if errorlevel 1 (
    echo.
    echo The Forest manual-label batch 5 GUI exited with an error.
    pause
)

endlocal
