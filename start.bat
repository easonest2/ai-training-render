@echo off
REM Start Flask web service
start cmd /k "file.py"

REM Run the other batch file (replace other_script.bat with your file name)
start cmd /k "other_script.bat"

REM Keep window open
pause
