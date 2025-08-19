@echo off
chcp 65001 >nul
title MicroGPT Pro - Custom Training

echo.
echo ========================================
echo    MicroGPT Pro - Custom Training
echo ========================================
echo.
echo This script adds NEW content to your existing training
echo WITHOUT losing any previous progress!
echo.
echo Choose your training option:
echo.
echo 1. Type custom training prompts (interactive)
echo 2. Use training data from text file
echo 3. Use default training_data folder
echo 4. Exit
echo.

:menu
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto custom_prompts
if "%choice%"=="2" goto text_file
if "%choice%"=="3" goto default_data
if "%choice%"=="4" goto exit
echo Invalid choice. Please enter 1, 2, 3, or 4.
echo.
goto menu

:custom_prompts
echo.
echo ========================================
echo    Custom Training Prompts
echo ========================================
echo.
echo This will create a file with your custom prompts
echo and add them to your existing model training.
echo.
echo Starting custom prompts training...
python custom_train.py --custom_prompts
echo.
pause
goto menu

:text_file
echo.
echo ========================================
echo    Text File Training
echo ========================================
echo.
echo This will add content from a text file
echo to your existing model training.
echo.
set /p file_path="Enter path to text file: "
if "%file_path%"=="" (
    echo No file path entered. Returning to menu.
    pause
    goto menu
)
echo.
echo Starting training with text file...
python custom_train.py --data_file "%file_path%"
echo.
pause
goto menu

:default_data
echo.
echo ========================================
echo    Default Training Data
echo ========================================
echo.
echo This will use the training_data folder
echo to add to your existing model training.
echo.
echo Starting training with default data...
python custom_train.py --data_dir training_data/
echo.
pause
goto menu

:exit
echo.
echo Thank you for using MicroGPT Pro Custom Training!
echo Your training progress is automatically saved in checkpoints.
echo You can run this anytime to add more content!
echo.
pause
exit
