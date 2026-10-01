@echo off
setlocal
cd /d "%~dp0\..\.."
echo timePassageOne - 50 regularly spaced frames.
echo Ctrl+1 = mob   Ctrl+2 = hero
echo 1-25: timePassageOneSample0   26-50: timePassageOneSample1
echo All 50 frames have been reviewed. Canonical labels are preserved.
echo CPU training is complete; results are in experiments/pretrained-v1-20260930.
echo Frames 42-43 are hero-only event scenes. Check frame 45's heavily hidden hero.
echo Click the Frame number, type a position, and press Enter to jump.
python perception_cli.py --map timePassageOne monster-review --images-only --split pilot --queue all
if errorlevel 1 (
    echo timePassageOne review exited with an error.
    pause
)
endlocal
