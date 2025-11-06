"""
Quick script to retrain LSTM and CNN models with corrected input shapes.

This script:
1. Loads the data correctly (vocal for LSTM, accompaniment for CNN)
2. Prepares data with correct shapes: LSTM (132, 40), CNN (40, 132, 1)
3. Trains both models
4. Saves them to the models directory
"""

import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv2D, MaxPooling2D, 
    Bidirectional, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths
DATA_DIR = Path("Data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Load data
print("Loading data...")
with open(DATA_DIR / "data.json", 'r') as f:
    vocal_data = json.load(f)

with open(DATA_DIR / "accompaniment_mfcc.json", 'r') as f:
    accomp_data = json.load(f)

# Extract features and labels
X_vocal = np.array(vocal_data['mfcc'])
X_accomp = np.array(accomp_data['mfcc'])
y = np.array(vocal_data['labels'])

print(f"Original shapes:")
print(f"X_vocal: {X_vocal.shape}")
print(f"X_accomp: {X_accomp.shape}")

# Prepare data for LSTM: (samples, time_steps=132, features=40)
def prepare_mfcc_for_lstm(X):
    """Prepare MFCCs for LSTM input: (samples, 132, 40)"""
    X_processed = X.copy()
    
    # Expected input: (samples, 40, 132)
    if X_processed.shape[1:] == (40, 132):
        # Transpose to (samples, 132, 40) - (batch, time_steps, features)
        X_processed = np.transpose(X_processed, (0, 2, 1))
    
    print(f"LSTM processed shape: {X_processed.shape}")
    return X_processed

# Prepare data for CNN: (samples, 40, 132, 1)
def prepare_mfcc_for_cnn(X):
    """Prepare MFCCs for CNN input: (samples, 40, 132, 1)"""
    X_processed = X.copy()
    
    # Add channel dimension
    X_processed = X_processed[..., np.newaxis]
    
    print(f"CNN processed shape: {X_processed.shape}")
    return X_processed

# Process data
X_vocal_processed = prepare_mfcc_for_lstm(X_vocal)
X_accomp_processed = prepare_mfcc_for_cnn(X_accomp)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
with open(MODELS_DIR / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"\nLabel encoder saved. Classes: {label_encoder.classes_}")

# Split data (60% train, 20% validation, 20% test)
X_vocal_train, X_vocal_temp, y_train, y_temp = train_test_split(
    X_vocal_processed, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
)
X_vocal_val, X_vocal_test, y_val, y_test = train_test_split(
    X_vocal_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_accomp_train, X_accomp_temp, _, _ = train_test_split(
    X_accomp_processed, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
)
X_accomp_val, X_accomp_test, _, _ = train_test_split(
    X_accomp_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nData split:")
print(f"Train: {len(X_vocal_train)} samples")
print(f"Val: {len(X_vocal_val)} samples")
print(f"Test: {len(X_vocal_test)} samples")

# Build LSTM model
def build_lstm_model(input_shape=(132, 40), num_classes=10):
    """Build LSTM model for vocal features"""
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(256)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Build CNN model
def build_cnn_model(input_shape=(40, 132, 1), num_classes=10):
    """Build CNN model for accompaniment features"""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train LSTM model
print("\n" + "="*50)
print("Training LSTM model for vocal features...")
print("="*50)

lstm_model = build_lstm_model(input_shape=(132, 40), num_classes=len(label_encoder.classes_))
lstm_model.summary()

lstm_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

lstm_history = lstm_model.fit(
    X_vocal_train, y_train,
    validation_data=(X_vocal_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=lstm_callbacks,
    verbose=1
)

# Evaluate LSTM
lstm_loss, lstm_acc = lstm_model.evaluate(X_vocal_test, y_test, verbose=0)
print(f"\nLSTM Test Accuracy: {lstm_acc:.4f}")

# Save LSTM model
lstm_model.save(MODELS_DIR / "lstm_vocal_classifier.keras")
print(f"✅ LSTM model saved to {MODELS_DIR / 'lstm_vocal_classifier.keras'}")

# Train CNN model
print("\n" + "="*50)
print("Training CNN model for accompaniment features...")
print("="*50)

cnn_model = build_cnn_model(input_shape=(40, 132, 1), num_classes=len(label_encoder.classes_))
cnn_model.summary()

cnn_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

cnn_history = cnn_model.fit(
    X_accomp_train, y_train,
    validation_data=(X_accomp_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=cnn_callbacks,
    verbose=1
)

# Evaluate CNN
cnn_loss, cnn_acc = cnn_model.evaluate(X_accomp_test, y_test, verbose=0)
print(f"\nCNN Test Accuracy: {cnn_acc:.4f}")

# Save CNN model
cnn_model.save(MODELS_DIR / "cnn_accompaniment_classifier.keras")
print(f"✅ CNN model saved to {MODELS_DIR / 'cnn_accompaniment_classifier.keras'}")

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"LSTM Test Accuracy: {lstm_acc:.4f}")
print(f"CNN Test Accuracy: {cnn_acc:.4f}")
print(f"\nModels saved to {MODELS_DIR}/")
print("\nNext steps:")
print("1. Run save_meta_models.py to train meta-models")
print("2. Test with Streamlit app: .\\run_streamlit.bat")
