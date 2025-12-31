@echo off
echo ========================================
echo RVC TTS WebUI - PyTorch Fix Script
echo ========================================
echo.

echo [1/5] Uninstalling old PyTorch...
pip uninstall torch torchaudio torchvision -y
echo.

echo [2/5] Clearing pip cache...
pip cache purge
echo.

echo [3/5] Installing PyTorch 2.0.1 (CPU version)...
pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
echo.

echo [4/5] Installing other dependencies...
pip install -r requirements.txt
echo.

echo [5/5] Verifying installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo.

echo ========================================
echo Installation complete!
echo You can now run: python app.py
echo ========================================
pause
