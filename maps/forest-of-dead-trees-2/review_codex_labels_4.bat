@echo off
setlocal
cd /d "%~dp0\..\.."

echo.
echo FOREST MANUAL LABELING - BATCH 4 ^(15 FRAMES^)
echo ------------------------------------------------------------
echo Your existing manual boxes were preserved.
echo Later frames now contain Codex pre-labels for you to inspect and correct.
echo Confirm that every visible Zombie, Hero, and Lich is covered exactly once.
echo.
echo  Ctrl+1 = Zombie     Ctrl+2 = Hero     Ctrl+3 = Lich
echo  After choosing a type, click the ground/feet position to add it.
echo  Drag a box to move it. Del removes the selected box.
echo  Enter confirms the frame and moves to the next one.
echo  Left/Right revisits frames. Q saves and quits. Window X also saves.
echo ------------------------------------------------------------
echo.

python perception_cli.py --map forest-of-dead-trees-2 monster-review ^
    --video "C:\projects\input-flag-inspector\saves_m\ForestOfDeadTreeSample4_LottaLich\screen.mp4" ^
    --split all ^
    --queue all ^
    --frame-id-prefix codex-sample4-hardcase- ^
    --start-id codex-sample4-hardcase-0002

if errorlevel 1 (
    echo.
    echo The Forest manual-label batch 4 GUI exited with an error.
    pause
)

endlocal
