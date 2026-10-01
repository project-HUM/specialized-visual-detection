@echo off
setlocal
cd /d "%~dp0\..\.."
".venv\Scripts\python.exe" "maps\timePassageOne\dataset\runs\pretrained-v1-20260930\runner.py" train --name pretrained-640 --resume
if errorlevel 1 pause
endlocal
