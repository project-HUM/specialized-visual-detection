@echo off
setlocal
cd /d "%~dp0\..\.."

echo.
echo FOREST MANUAL LABELING - BATCH 6 ^(29 HUMAN EVIDENCE FRAMES^)
echo ------------------------------------------------------------
echo 17 Lich-positive frames and 12 hard negatives, all pending review.
echo 23 training / 6 validation frames, grouped by source HUMAN run.
echo Keep the assigned splits: nearby frames must not cross splits.
echo Inspect every Zombie, Hero and Lich, including hidden/clipped sprites.
echo Signs, terrain, loot, pets and UI pictures are not Lich labels.
echo Full preset boxes include estimated hidden feet; correct as needed.
echo Still images only; no temporal video context is available.
echo.
echo Ctrl+1 Zombie / Ctrl+2 Hero / Ctrl+3 Lich
echo Click feet to add. Drag to move. Del removes. Enter confirms.
echo Left/Right revisits frames. Q or Window X saves and quits.
echo ------------------------------------------------------------
echo.

python perception_cli.py --map forest-of-dead-trees-2 monster-review ^
    --images-only ^
    --split all ^
    --queue all ^
    --frame-id-prefix codex-human-batch6-evidence- ^
    --start-id codex-human-batch6-evidence-0000

if errorlevel 1 (
    echo.
    echo The Forest manual-label batch 6 GUI exited with an error.
    pause
)
endlocal
