# How to Download Large Data Files

## Problem
The data files (`data.json` and `accompaniment_mfcc.json`) are stored using Git LFS and haven't been downloaded yet. You need them to retrain the models.

## Solution

### Option 1: Download with Git LFS (Recommended)

1. **Install Git LFS** (if not already installed):
   ```powershell
   # Download from: https://git-lfs.github.com/
   # Or install with Chocolatey:
   choco install git-lfs
   ```

2. **Initialize Git LFS**:
   ```powershell
   git lfs install
   ```

3. **Pull the large files**:
   ```powershell
   git lfs pull
   ```

### Option 2: Use Existing Trained Models

If you don't want to retrain, you can use the existing models but they have the **wrong input shape**. 

**The current models expect**: `(40, 132)` for LSTM  
**The code now expects**: `(132, 40)` for LSTM  

So you'll get the dimension mismatch error you're seeing.

### Option 3: Generate New Training Data

If the original data is too large or unavailable, you could:

1. Run the preprocessing notebooks in `data_analysis/` to regenerate the JSON files from audio
2. Make sure the audio files are available in `Data/genres_original/`

## Why This Happened

The Git repository uses LFS for large files (>100MB). When you clone the repo:
- Small text files are downloaded normally
- Large files show as LFS pointers until you run `git lfs pull`

## File Sizes

- `data.json`: ~662 MB (MFCC features for vocal)
- `accompaniment_mfcc.json`: ~662 MB (MFCC features for accompaniment)

## Next Steps After Downloading

Once you have the data files:

1. **Retrain the base models**:
   ```powershell
   python retrain_base_models.py
   ```
   This will take ~30-60 minutes depending on your hardware.

2. **Train the meta-models**:
   ```powershell
   python save_meta_models.py
   ```

3. **Test the Streamlit app**:
   ```powershell
   .\run_streamlit.bat
   ```

## Alternative: Quick Fix for Testing

If you just want to test the app quickly without retraining, I can modify the code to handle both old and new model formats. However, this is a workaround and not ideal for production use.
