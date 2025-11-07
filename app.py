import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import json
import os
from pathlib import Path

# Model files mapping
files = {
    'DNN-57.51%': 'models/model_dnn.h5',
    'CNN(without regularization)-69.69%': 'models/model_cnn1.h5',
    'CNN(with regularization)-78.6%': 'models/model_cnn2.h5',
    'CNN(with regularization and data augmentation)-83.71%': 'models/model_cnn3.h5',
    'FCNN(without data slicing)-92.12%': 'models/fcnn_melspec_gtzan.h5',
    'FCNN(with data slicing)-75%': 'models/fcnn_splice_songs_model.h5',
    'FCNN(with data augmentation)-75%': 'models/fcnn_genre_classification_data_aug_model.h5',
    'Short chunk CNN-81.4%': 'models/short_chunk_cnn.keras',
    'MusiCNN-83%': 'models/musicnn.keras',
    'CRNN-83%': 'models/crnn.keras',
    'Noisy student training-83.38%': 'models/student_nst.keras',
    'Knowledge expansion & distillation-81.64%': 'models/ked_student_nst.keras',
    'LSTM(vocal)-63.75%': 'models/lstm_vocal_classifier.keras',
    'CNN(accompaniment)-74.7%': 'models/cnn_accompaniment_classifier.keras'
}

# Genre mapping
genre_map = {
    "0": "blues",
    "1": "classical",
    "2": "country",
    "3": "disco",
    "4": "hiphop",
    "5": "jazz",
    "6": "metal",
    "7": "pop",
    "8": "reggae",
    "9": "rock"
}
genres = [genre_map[str(i)] for i in range(len(genre_map))]

# Function to extract MFCC features from audio
def extract_mfcc(file_path, duration=30, n_mfcc=13, sr=22050, hop_length=512):
    try:
        audio, sr = librosa.load(file_path, duration=duration, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        max_length = 130  # Match model input shape
        if mfcc.shape[1] > max_length:
            mfcc = mfcc[:, :max_length]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
        mfcc = mfcc.T[np.newaxis, :, :, np.newaxis]  # (1, 130, 13, 1)
        return mfcc
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Streamlit app
st.set_page_config(page_title="Music Genre Classification", page_icon="🎵", layout="wide")
st.title("🎵 Music Genre Classification")
st.markdown("Upload an audio file to predict its genre using various deep learning models.")

learning = st.selectbox("Choose the learning type", options=["Supervised learning", "Semi-Supervised learning"], index=0)

if learning == "Supervised learning":
    model_type = st.selectbox("Choose a model", options=[
        "DNN-57.51%",
        "CNN(without regularization)-69.69%",
        "CNN(with regularization)-78.6%",
        "CNN(with regularization and data augmentation)-83.71%",
        "FCNN(without data slicing)-92.12%",
        "FCNN(with data slicing)-75%",
        "FCNN(with data augmentation)-75%",
        "Short chunk CNN-81.4%",
        "MusiCNN-83%",
        "CRNN-83%"
    ], index=0)
elif learning == "Semi-Supervised learning":
    model_type = st.selectbox("Choose a model", options=[
        "Noisy student training-83.38%",
        "Knowledge expansion & distillation-81.64%"
    ], index=0)
elif learning == "Ensemble learning":
    model_type = st.selectbox("Choose a model", options=[
        "LSTM(vocal)-63.75%",
        "CNN(accompaniment)-74.7%"
    ], index=0)
else:
    model_type = None

# Load model
if model_type:
    model_path = files[model_type]
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    try:
        with st.spinner(f"Loading {model_type}..."):
            model = tf.keras.models.load_model(model_path)
        st.success(f"✅ Model loaded: {model_type}")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
else:
    st.error("❌ Please select a model")
    st.stop()

st.write("Upload an audio file (WAV or MP3) to predict its genre.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    temp_audio_path = "temp_audio.wav"
    
    try:
        # Save uploaded file
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(uploaded_file, format="audio/wav")
        
        # Extract features and predict
        with st.spinner("Extracting features and making prediction..."):
            mfcc = extract_mfcc(temp_audio_path)
        
        if mfcc is not None:
            prediction = model.predict(mfcc)
            predicted_genre_idx = np.argmax(prediction, axis=1)[0]
            predicted_genre = genres[predicted_genre_idx]
            probabilities = prediction[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"### Predicted Genre: **{predicted_genre}**")
                st.info("💡 For better results, try different models!")
            
            with col2:
                st.write("**Top 3 Predictions:**")
                top_3_idx = np.argsort(probabilities)[-3:][::-1]
                for idx in top_3_idx:
                    st.write(f"- {genres[idx]}: {probabilities[idx]*100:.2f}%")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass