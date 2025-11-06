import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import json

files={'DNN-57.51%': 'models/model_dnn.h5','CNN(without regularization)-69.69%': 'models/model_cnn1.h5','CNN(with regularization)-78.6%': 'models/model_cnn2.h5','CNN(with regularization and data augmentation)-83.71%': 'models/model_cnn3.h5','FCNN(without data slicing)-92.12%': 'models/fcnn_melspec_gtzan.h5','FCNN(with data slicing)-75%': 'models/fcnn_splice_songs_model.h5','FCNN(with data augmentation)-75%': 'models/fcnn_genre_classification_data_aug_model.h5','Short chunk CNN-81.4%': 'models/short_chunk_cnn.keras','MusiCNN-83%': 'models/musicnn.keras','CRNN-83%': 'models/crnn.keras','Noisy student training-83.38%': 'models/student_nst.keras','Knowledge expansion & distillation-81.64%': 'models/ked_student_nst.keras','LSTM(vocal)-63.75%': 'models/lstm_vocal_classifier.keras','CNN(accompaniment)-74.7%': 'models/cnn_accompaniment_classifier.keras'}


# Load genre mapping
with open('genres.json', 'r') as fp:
    genre_map = json.load(fp)
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
st.title("Music Genre Classification")
learning = st.selectbox("Choose the learning type", options=["Supervised learning", "Semi-Supervised learning"], index=0)

if learning == "Supervised learning":
    model_type = st.selectbox("Choose a model", options=["DNN-57.51%","CNN(without regularization)-69.69%","CNN(with regularization)-78.6%","CNN(with regularization and data augmentation)-83.71%","FCNN(without data slicing)-92.12%","FCNN(with data slicing)-75%","FCNN(with data augmentation)-75%", "Short chunk CNN-81.4%","MusiCNN-83%","CRNN-83%"], index=0)
elif learning == "Semi-Supervised learning":
    model_type = st.selectbox("Choose a model", options=["Noisy student training-83.38%","Knowledge expansion & distillation-81.64%"], index=0)
elif learning == "Ensemble learning":
    model_type = st.selectbox("Choose a model", options=["LSTM(vocal)-63.75%", "CNN(accompaniment)-74.7%"], index=0)

model = tf.keras.models.load_model(files[model_type])

st.write("Upload an audio file (WAV or MP3) to predict its genre.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    mfcc = extract_mfcc("temp_audio.wav")
    
    if mfcc is not None:
        prediction = model.predict(mfcc)
        predicted_genre_idx = np.argmax(prediction, axis=1)[0]
        predicted_genre = genres[predicted_genre_idx]
        probabilities = prediction[0]
        
        st.success(f"Predicted Genre: **{predicted_genre}**")
        st.success("!! For better results, please check with the other models as well !!")

# Clean up
import os
if os.path.exists("temp_audio.wav"):
    os.remove("temp_audio.wav")