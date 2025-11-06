@echo off
REM Quick start script for Streamlit Ensemble App
REM Windows batch file

echo ========================================
echo Music Genre Classifier - Streamlit App
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Dependencies not installed. Installing now...
    echo.
    pip install -r requirements_streamlit.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Checking models...
if not exist "models\lstm_vocal_classifier.keras" (
    echo WARNING: LSTM model not found at models\lstm_vocal_classifier.keras
    echo Please train the models first using Ensemble_learning_FIXED.ipynb
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "!continue!"=="y" exit /b 1
) else (
    echo LSTM model found
)

if not exist "models\cnn_accompaniment_classifier.keras" (
    echo WARNING: CNN model not found at models\cnn_accompaniment_classifier.keras
    echo Please train the models first using Ensemble_learning_FIXED.ipynb
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "!continue!"=="y" exit /b 1
) else (
    echo CNN model found
)

echo.
echo [3/3] Starting Streamlit app...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

streamlit run streamlit_ensemble_app.py

pause
