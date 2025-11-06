# Music Genre Classification - Enhanced with Streamlit

This project implements music genre classification using deep learning with an **improved ensemble learning approach** and an **interactive Streamlit web application**.

## 🎵 What's New

### ✅ Fixed Ensemble Implementation
- Separate vocal and accompaniment feature extraction
- Proper train/validation/test splits (no data leakage)
- Multiple ensemble strategies (voting + stacking)
- Clean, production-ready code

### 🌟 Streamlit Web App
- **Upload audio files** and get instant genre predictions
- **Interactive visualizations** of prediction probabilities
- **6 ensemble methods** to choose from
- **Compare models** side-by-side (LSTM vs CNN vs Ensemble)

## 🚀 Quick Start

### Option 1: Run Streamlit App (Recommended)

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run streamlit_ensemble_app.py
```

Or on Windows, double-click `run_streamlit.bat`

### Option 2: Train Models from Scratch

1. Open `ensemble/Ensemble_learning_FIXED.ipynb` in Jupyter
2. Run all cells to train models
3. Generate meta-models: `python save_meta_models.py`
4. Launch Streamlit app: `streamlit run streamlit_ensemble_app.py`

## 🎯 Supported Genres

Blues • Classical • Country • Disco • Hip-hop • Jazz • Metal • Pop • Reggae • Rock

## 🏆 Model Performance

- **LSTM (Vocal)**: ~72-75% accuracy
- **CNN (Accompaniment)**: ~75-77% accuracy
- **XGBoost Stacking**: ~79-82% accuracy ⭐ **Best**

## 📖 Documentation

- **[STREAMLIT_README.md](STREAMLIT_README.md)** - Complete Streamlit app guide
- **[ensemble/README_FIXES.md](ensemble/README_FIXES.md)** - Detailed fixes documentation

## 💡 Key Improvements

### What Was Fixed

1. **Data Leakage** ❌ → ✅ - Proper train/val/test split
2. **Data Mismatch** ❌ → ✅ - Separate vocal/accompaniment features
3. **Architecture Issues** ❌ → ✅ - Clean, consistent models
4. **No Validation** ❌ → ✅ - Proper 60/20/20 split

## 🛠️ Technologies

TensorFlow • Keras • XGBoost • Streamlit • Plotly • librosa

---

**Made with ❤️ and 🎵 | Star ⭐ this repo if you find it helpful!**
