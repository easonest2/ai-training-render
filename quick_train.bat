@echo off
echo MicroGPT Pro - Quick Training Start
echo ===================================
echo.
echo This script will:
echo 1. Create sample training data
echo 2. Start training your model
echo 3. Save the trained model automatically
echo.
pause

echo.
echo Step 1: Loading data...


echo.
echo Step 2: Starting training...
echo Training will take 5-15 minutes on CPU, 1-3 minutes on GPU
echo.
python train.py --data training_data/combined_training.txt --epochs 5

echo.
echo Training completed! Your model is saved in artifacts/model_final.pt
echo.
echo To use your trained model:
echo 1. Restart the server: python serve_memory.py
echo 2. Test it: python demo.py
echo.
pause
