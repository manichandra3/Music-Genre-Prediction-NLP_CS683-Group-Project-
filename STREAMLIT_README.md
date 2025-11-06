# Streamlit Ensemble App - Music Genre Classification

## 🎵 Overview

This is a web application built with Streamlit that provides an interactive interface for music genre classification using the **fixed ensemble learning implementation**. It combines LSTM and CNN models with multiple ensemble strategies.

## ✨ Features

- **Real-time audio classification** - Upload any audio file and get instant genre predictions
- **Multiple ensemble methods** - Choose from 6 different ensemble strategies:
  - Mean Averaging
  - Weighted Voting (Recommended)
  - Max Voting
  - Logistic Regression Stacking
  - XGBoost Stacking (Best Performance)
  - Neural Network Stacking
- **Interactive visualizations** - Beautiful charts showing prediction probabilities
- **Model comparison** - Compare LSTM, CNN, and ensemble predictions side-by-side
- **Confidence scores** - See how confident the model is in its predictions

## 📋 Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

Or install manually:
```bash
pip install streamlit tensorflow numpy scikit-learn xgboost librosa plotly pandas
```

### 2. Train Models

You must have trained models in the `models/` directory:

**Required:**
- `models/lstm_vocal_classifier.keras` - LSTM model for vocal features
- `models/cnn_accompaniment_classifier.keras` - CNN model for accompaniment features

**Optional (for stacking methods):**
- `models/lr_meta_model.pkl` - Logistic Regression meta-model
- `models/xgb_meta_model.pkl` - XGBoost meta-model
- `models/nn_meta_model.keras` - Neural Network meta-model

### 3. Generate Meta-Models

If you have trained the base models (LSTM and CNN), run:

```bash
python save_meta_models.py
```

This will:
- Load your trained LSTM and CNN models
- Generate predictions on the validation set
- Train and save the meta-models for stacking

## 🚀 Running the App

### Start the Streamlit App

```bash
streamlit run streamlit_ensemble_app.py
```

The app will open in your browser at `http://localhost:8501`

### Alternative: Specify Port

```bash
streamlit run streamlit_ensemble_app.py --server.port 8080
```

## 📖 How to Use

### Step 1: Launch the App
Run the command above and wait for your browser to open.

### Step 2: Select Ensemble Method
In the sidebar, choose your preferred ensemble method:
- **Weighted Voting** - Good default choice, adapts to each prediction
- **XGBoost Stacking** - Best overall performance (requires trained meta-model)
- Other methods available based on trained models

### Step 3: Upload Audio
Click "Browse files" and select an audio file:
- Supported formats: WAV, MP3, OGG, FLAC, M4A
- Best results with 30-second clips
- High-quality audio recommended

### Step 4: Classify
Click "🎯 Classify Genre" and wait for results!

### Step 5: Explore Results
- See the predicted genre with confidence score
- View probability distributions for all genres
- Compare predictions across LSTM, CNN, and ensemble
- Check top 3 predictions

## 🎯 Supported Genres

1. Blues
2. Classical
3. Country
4. Disco
5. Hip-hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock

## 🏗️ Architecture

### Model Pipeline

```
Audio File
    ↓
MFCC Extraction (40 x 132)
    ↓
    ├──→ LSTM Model (Vocal) ──┐
    │                          │
    └──→ CNN Model (Accomp) ───┼──→ Ensemble Methods
                                │
                                ├──→ Voting (Mean/Weighted/Max)
                                │
                                └──→ Stacking (LR/XGBoost/NN)
                                    ↓
                               Genre Prediction
```

### Ensemble Methods

| Method | Type | Pros | Cons |
|--------|------|------|------|
| **Mean Averaging** | Voting | Simple, fast | No adaptation |
| **Weighted Voting** | Voting | Adaptive, no training | Local weights |
| **Max Voting** | Voting | Picks confident predictions | Can be unstable |
| **LR Stacking** | Meta-learning | Fast, interpretable | Linear only |
| **XGBoost Stacking** | Meta-learning | Best performance | Requires training |
| **NN Stacking** | Meta-learning | Non-linear | Complex, slower |

## 📊 Model Information

### LSTM Model (Vocal Features)
```
Input: (40, 132)
- Bidirectional LSTM layers with 256 units
- Dense layers: 256 → 128 → 32 → 10
- Dropout: 0.5
- Activation: Softmax
```

### CNN Model (Accompaniment Features)
```
Input: (40, 132, 1)
- Conv2D layers: 64 → 128 → 256 → 512
- MaxPooling after each conv block
- Dense layers: 256 → 128 → 10
- Dropout: 0.5
- Activation: Softmax
```

## 🔧 Configuration

### Adjusting Parameters

Edit `streamlit_ensemble_app.py` to modify:

```python
# Audio parameters
SAMPLE_RATE = 22050  # Hz
DURATION = 30  # seconds
N_MFCC = 40  # Number of MFCC coefficients
N_FRAMES = 132  # Number of time frames

# Model paths
MODELS_DIR = Path("models")
LSTM_MODEL_PATH = MODELS_DIR / "lstm_vocal_classifier.keras"
CNN_MODEL_PATH = MODELS_DIR / "cnn_accompaniment_classifier.keras"
```

## 🐛 Troubleshooting

### "No models loaded" Error
**Problem:** Model files not found

**Solution:**
```bash
# Check models directory
ls models/

# You should see:
# lstm_vocal_classifier.keras
# cnn_accompaniment_classifier.keras
```

Train models using `Ensemble_learning_FIXED.ipynb` if missing.

### "Error extracting features" Error
**Problem:** Audio file format issue

**Solution:**
- Convert audio to WAV format
- Ensure audio file is not corrupted
- Try a different audio file

### Stacking Methods Not Available
**Problem:** Meta-models not trained

**Solution:**
```bash
python save_meta_models.py
```

This trains and saves meta-models.

### Low Memory Error
**Problem:** Not enough RAM for large models

**Solution:**
- Close other applications
- Use voting methods instead of stacking
- Reduce model size in training

## 📈 Performance Tips

### For Best Results:
1. **Use high-quality audio** - Clear recordings work best
2. **30-second clips** - Optimal duration for feature extraction
3. **Try multiple methods** - Compare results across ensemble strategies
4. **XGBoost stacking** - Usually provides best accuracy

### Speed Optimization:
1. **GPU acceleration** - Ensure TensorFlow can use your GPU
2. **Model caching** - Models are loaded once and cached
3. **Batch processing** - Process multiple files in sequence

## 📁 Project Structure

```
Music-Genre-Classification-main/
├── streamlit_ensemble_app.py      # Main Streamlit application
├── save_meta_models.py            # Script to train meta-models
├── requirements_streamlit.txt     # Python dependencies
├── ensemble/
│   ├── Ensemble_learning_FIXED.ipynb  # Training notebook
│   └── README_FIXES.md            # Detailed fixes documentation
├── models/
│   ├── lstm_vocal_classifier.keras
│   ├── cnn_accompaniment_classifier.keras
│   ├── lr_meta_model.pkl          # (optional)
│   ├── xgb_meta_model.pkl         # (optional)
│   └── nn_meta_model.keras        # (optional)
└── Data/
    ├── data.json                   # Vocal MFCCs
    └── accompaniment_mfcc.json     # Accompaniment MFCCs
```

## 🔬 Technical Details

### Feature Extraction
- **Method:** Mel-Frequency Cepstral Coefficients (MFCC)
- **Coefficients:** 40
- **Time frames:** 132
- **Sample rate:** 22050 Hz
- **Duration:** 30 seconds

### Ensemble Strategy
The app uses **proper ensemble methodology**:
1. Base models trained on training set
2. Meta-models trained on validation predictions
3. Final evaluation on held-out test set
4. No data leakage ✅

## 📚 Additional Resources

- [Original Notebook](ensemble/Ensemble_learning_FIXED.ipynb) - Complete training pipeline
- [Fix Documentation](ensemble/README_FIXES.md) - Detailed fixes and improvements
- [Streamlit Docs](https://docs.streamlit.io/) - Streamlit documentation
- [TensorFlow Docs](https://www.tensorflow.org/) - TensorFlow documentation

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to:
1. Open an issue
2. Submit a pull request
3. Improve documentation

## 📝 License

Same license as the parent project.

## 🎉 Acknowledgments

- Fixed ensemble implementation based on proper ML methodology
- Uses separate vocal and accompaniment features
- Implements multiple ensemble strategies
- Built with Streamlit for easy deployment

---

**Happy Classifying! 🎵**
