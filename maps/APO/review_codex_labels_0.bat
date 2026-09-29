@echo off
setlocal
cd /d "%~dp0\..\.."
echo APO - 80 frames in saved source groups; original 50 stay in place.
echo Ctrl+1 = mob   Ctrl+2 = hero   Ctrl+3 = special
echo All 80 frames have been reviewed; this launcher can revisit any frame.
echo 1-50: sample2   51-65: sample1   66-80: sample3_woSpecial
echo The ghost-like NPC with a book thought bubble is not a mob target.
echo Click the Frame number, type a position, and press Enter to jump.
echo Frame 66 is a sampled town image; only the hero is labeled.
python perception_cli.py --map APO monster-review --images-only --split pilot --queue all
if errorlevel 1 (
    echo APO review exited with an error.
    pause
)
endlocal
