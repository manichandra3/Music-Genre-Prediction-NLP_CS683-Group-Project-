"""
Save trained meta-models for Streamlit app

This script should be run after training the ensemble models in the notebook.
It saves the meta-models (Logistic Regression, XGBoost, Neural Network) 
in pickle/keras format for use in the Streamlit application.
"""

import tensorflow as tf
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
DATA_PATH_VOCAL = 'Data/data.json'
DATA_PATH_ACCOMP = 'Data/accompaniment_mfcc.json'
LSTM_MODEL_PATH = 'models/lstm_vocal_classifier.keras'
CNN_MODEL_PATH = 'models/cnn_accompaniment_classifier.keras'

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_and_prepare_data():
    """Load and prepare data for meta-model training"""
    print("Loading data...")
    
    # Load vocal data
    with open(DATA_PATH_VOCAL, 'r', encoding='utf-8') as f:
        data_vocal = json.load(f)
    
    # Load accompaniment data
    try:
        with open(DATA_PATH_ACCOMP, 'r', encoding='utf-8') as f:
            data_accomp = json.load(f)
        print("✓ Loaded separate vocal and accompaniment data")
    except FileNotFoundError:
        print("⚠ Warning: Using vocal data for both (accompaniment data not found)")
        data_accomp = data_vocal
    
    # Extract features
    mfccs_vocal = data_vocal['mfcc']
    mfccs_accomp = data_accomp['mfcc']
    labels = data_vocal['genre_num']
    
    # Align data
    min_length = min(len(mfccs_vocal), len(mfccs_accomp), len(labels))
    mfccs_vocal = mfccs_vocal[:min_length]
    mfccs_accomp = mfccs_accomp[:min_length]
    labels = labels[:min_length]
    
    # Convert to numpy
    X_vocal = np.array(mfccs_vocal)
    X_accomp = np.array(mfccs_accomp)
    y = np.array(labels)
    
    # Prepare for LSTM: (samples, 132, 40) - (batch, time_steps, features)
    if X_vocal.shape[1:] == (130, 13):
        X_vocal = X_vocal[:, :40, :]  # Take first 40 coefficients
        X_vocal = np.pad(X_vocal, ((0, 0), (0, 0), (0, 132 - X_vocal.shape[2])), mode='constant')
        # Transpose to (samples, time_steps, features)
        X_vocal = np.transpose(X_vocal, (0, 2, 1))  # (samples, 132, 40)
    elif X_vocal.shape[1:] == (40, 132):
        # Already correct shape, just transpose
        X_vocal = np.transpose(X_vocal, (0, 2, 1))  # (samples, 132, 40)
    
    # Prepare for CNN: (samples, 40, 132, 1)
    if X_accomp.shape[1:] == (130, 13):
        X_accomp = X_accomp[:, :40, :]  # Take first 40 coefficients
        X_accomp = np.pad(X_accomp, ((0, 0), (0, 0), (0, 132 - X_accomp.shape[2])), mode='constant')
        # Transpose to (samples, 40, 132)
        X_accomp = np.transpose(X_accomp, (0, 1, 2))  # Keep as is
    X_accomp = X_accomp[..., np.newaxis]  # Add channel dimension
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X_vocal, X_accomp, y_encoded

def split_data(X_vocal, X_accomp, y_encoded):
    """Split data into train/val/test sets"""
    print("Splitting data...")
    
    # First split: 60% train, 40% temp
    X_vocal_train, X_vocal_temp, X_accomp_train, X_accomp_temp, y_train, y_temp = train_test_split(
        X_vocal, X_accomp, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42
    )
    
    # Second split: 20% val, 20% test
    X_vocal_val, X_vocal_test, X_accomp_val, X_accomp_test, y_val, y_test = train_test_split(
        X_vocal_temp, X_accomp_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    return (X_vocal_train, X_accomp_train, y_train,
            X_vocal_val, X_accomp_val, y_val,
            X_vocal_test, X_accomp_test, y_test)

def generate_base_predictions(lstm_model, cnn_model, X_vocal_val, X_accomp_val):
    """Generate predictions from base models"""
    print("Generating base model predictions...")
    
    lstm_probs_val = lstm_model.predict(X_vocal_val, verbose=0)
    cnn_probs_val = cnn_model.predict(X_accomp_val, verbose=0)
    
    stacking_features_val = np.concatenate([lstm_probs_val, cnn_probs_val], axis=1)
    
    return stacking_features_val

def train_and_save_meta_models(stacking_features_val, y_val):
    """Train and save all meta-models"""
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression meta-model...")
    lr_meta = LogisticRegression(max_iter=1000, random_state=42)
    lr_meta.fit(stacking_features_val, y_val)
    
    lr_path = MODELS_DIR / 'lr_meta_model.pkl'
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_meta, f)
    print(f"✓ Saved to {lr_path}")
    
    # 2. XGBoost
    print("\nTraining XGBoost meta-model...")
    xgb_meta = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    xgb_meta.fit(stacking_features_val, y_val)
    
    xgb_path = MODELS_DIR / 'xgb_meta_model.pkl'
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_meta, f)
    print(f"✓ Saved to {xgb_path}")
    
    # 3. Neural Network
    print("\nTraining Neural Network meta-model...")
    
    # Further split for NN training
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        stacking_features_val, y_val, test_size=0.2, stratify=y_val, random_state=42
    )
    
    nn_meta = Sequential([
        Dense(128, activation='relu', input_dim=stacking_features_val.shape[1]),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(len(GENRES), activation='softmax')
    ])
    
    nn_meta.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    nn_meta.fit(
        X_meta_train, y_meta_train,
        validation_data=(X_meta_val, y_meta_val),
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )
    
    nn_path = MODELS_DIR / 'nn_meta_model.keras'
    nn_meta.save(nn_path)
    print(f"✓ Saved to {nn_path}")

def main():
    """Main execution function"""
    print("="*60)
    print("Meta-Model Training and Export Script")
    print("="*60)
    
    # Check if base models exist
    if not Path(LSTM_MODEL_PATH).exists():
        print(f"❌ Error: LSTM model not found at {LSTM_MODEL_PATH}")
        print("Please train the base models first using the notebook.")
        return
    
    if not Path(CNN_MODEL_PATH).exists():
        print(f"❌ Error: CNN model not found at {CNN_MODEL_PATH}")
        print("Please train the base models first using the notebook.")
        return
    
    print("✓ Base models found")
    
    # Load base models
    print("\nLoading base models...")
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print("✓ Base models loaded")
    
    # Load and prepare data
    X_vocal, X_accomp, y_encoded = load_and_prepare_data()
    
    # Split data
    (X_vocal_train, X_accomp_train, y_train,
     X_vocal_val, X_accomp_val, y_val,
     X_vocal_test, X_accomp_test, y_test) = split_data(X_vocal, X_accomp, y_encoded)
    
    print(f"\nData shapes:")
    print(f"  Train: {X_vocal_train.shape[0]} samples")
    print(f"  Val:   {X_vocal_val.shape[0]} samples")
    print(f"  Test:  {X_vocal_test.shape[0]} samples")
    
    # Generate base predictions
    stacking_features_val = generate_base_predictions(
        lstm_model, cnn_model, X_vocal_val, X_accomp_val
    )
    
    print(f"\nStacking features shape: {stacking_features_val.shape}")
    
    # Train and save meta-models
    train_and_save_meta_models(stacking_features_val, y_val)
    
    print("\n" + "="*60)
    print("✅ All meta-models trained and saved successfully!")
    print("="*60)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run streamlit_ensemble_app.py")

if __name__ == "__main__":
    main()
