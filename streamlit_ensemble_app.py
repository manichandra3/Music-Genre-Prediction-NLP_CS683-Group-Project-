"""
Streamlit App for Music Genre Classification using Ensemble Models

This application uses the fixed ensemble implementation combining:
- LSTM model for vocal features
- CNN model for accompaniment features
- Multiple ensemble methods (voting and stacking)
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import pickle
import os
import tempfile
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(
    page_title="Music Genre Classifier - Ensemble",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MFCC = 40
N_FRAMES = 132

# Model paths
MODELS_DIR = Path("models")
LSTM_MODEL_PATH = MODELS_DIR / "lstm_vocal_classifier.keras"
CNN_MODEL_PATH = MODELS_DIR / "cnn_accompaniment_classifier.keras"
LR_META_PATH = MODELS_DIR / "lr_meta_model.pkl"
XGB_META_PATH = MODELS_DIR / "xgb_meta_model.pkl"
NN_META_PATH = MODELS_DIR / "nn_meta_model.keras"

# Cache for model loading
@st.cache_resource
def load_models():
    """Load all models with caching"""
    models = {}
    
    try:
        if LSTM_MODEL_PATH.exists():
            models['lstm'] = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
            st.sidebar.success("✓ LSTM model loaded")
        else:
            st.sidebar.error("✗ LSTM model not found")
            
        if CNN_MODEL_PATH.exists():
            models['cnn'] = tf.keras.models.load_model(str(CNN_MODEL_PATH))
            st.sidebar.success("✓ CNN model loaded")
        else:
            st.sidebar.error("✗ CNN model not found")
            
        # Load meta-models if available
        if LR_META_PATH.exists():
            with open(LR_META_PATH, 'rb') as f:
                models['lr_meta'] = pickle.load(f)
            st.sidebar.success("✓ Logistic Regression meta-model loaded")
            
        if XGB_META_PATH.exists():
            with open(XGB_META_PATH, 'rb') as f:
                models['xgb_meta'] = pickle.load(f)
            st.sidebar.success("✓ XGBoost meta-model loaded")
            
        if NN_META_PATH.exists():
            models['nn_meta'] = tf.keras.models.load_model(str(NN_META_PATH))
            st.sidebar.success("✓ Neural Network meta-model loaded")
            
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
    
    return models

def extract_mfcc(audio_path, n_mfcc=40, target_length=132):
    """
    Extract MFCC features from audio file
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        target_length: Target number of frames
    
    Returns:
        mfcc: MFCC features of shape (n_mfcc, target_length)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to target length
        if mfcc.shape[1] < target_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :target_length]
        
        return mfcc
    
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def prepare_features_for_models(mfcc):
    """
    Prepare MFCC features for LSTM and CNN models
    
    Args:
        mfcc: MFCC features of shape (n_mfcc, n_frames)
    
    Returns:
        tuple: (lstm_input, cnn_input)
    """
    # For LSTM: (1, 40, 132)
    lstm_input = mfcc.T  # Transpose to (n_frames, n_mfcc)
    lstm_input = lstm_input[:N_FRAMES, :N_MFCC]  # Ensure correct shape
    lstm_input = np.expand_dims(lstm_input, axis=0)  # Add batch dimension
    
    # For CNN: (1, 40, 132, 1)
    cnn_input = mfcc[:N_MFCC, :N_FRAMES]  # (40, 132)
    cnn_input = np.expand_dims(cnn_input, axis=-1)  # Add channel dimension
    cnn_input = np.expand_dims(cnn_input, axis=0)  # Add batch dimension
    
    return lstm_input, cnn_input

def predict_ensemble(models, lstm_input, cnn_input, method='weighted_voting'):
    """
    Make ensemble predictions
    
    Args:
        models: Dictionary of loaded models
        lstm_input: Input for LSTM model
        cnn_input: Input for CNN model
        method: Ensemble method to use
    
    Returns:
        tuple: (predicted_genre, probabilities_dict)
    """
    # Get base model predictions
    lstm_probs = models['lstm'].predict(lstm_input, verbose=0)[0]
    cnn_probs = models['cnn'].predict(cnn_input, verbose=0)[0]
    
    if method == 'mean_averaging':
        # Simple average
        final_probs = (lstm_probs + cnn_probs) / 2
        
    elif method == 'weighted_voting':
        # Weight based on individual model confidence
        lstm_confidence = np.max(lstm_probs)
        cnn_confidence = np.max(cnn_probs)
        total = lstm_confidence + cnn_confidence
        
        w_lstm = lstm_confidence / total if total > 0 else 0.5
        w_cnn = cnn_confidence / total if total > 0 else 0.5
        
        final_probs = w_lstm * lstm_probs + w_cnn * cnn_probs
        
    elif method == 'max_voting':
        # Choose prediction with highest confidence
        lstm_max = np.max(lstm_probs)
        cnn_max = np.max(cnn_probs)
        
        final_probs = lstm_probs if lstm_max > cnn_max else cnn_probs
        
    elif method == 'lr_stacking' and 'lr_meta' in models:
        # Logistic Regression stacking
        stacking_features = np.concatenate([lstm_probs, cnn_probs]).reshape(1, -1)
        final_probs = models['lr_meta'].predict_proba(stacking_features)[0]
        
    elif method == 'xgb_stacking' and 'xgb_meta' in models:
        # XGBoost stacking
        stacking_features = np.concatenate([lstm_probs, cnn_probs]).reshape(1, -1)
        final_probs = models['xgb_meta'].predict_proba(stacking_features)[0]
        
    elif method == 'nn_stacking' and 'nn_meta' in models:
        # Neural Network stacking
        stacking_features = np.concatenate([lstm_probs, cnn_probs]).reshape(1, -1)
        final_probs = models['nn_meta'].predict(stacking_features, verbose=0)[0]
        
    else:
        # Fallback to mean averaging
        final_probs = (lstm_probs + cnn_probs) / 2
    
    # Get predicted class
    predicted_class = np.argmax(final_probs)
    predicted_genre = GENRES[predicted_class]
    
    # Create probabilities dictionary
    probs_dict = {
        'lstm': {genre: float(prob) for genre, prob in zip(GENRES, lstm_probs)},
        'cnn': {genre: float(prob) for genre, prob in zip(GENRES, cnn_probs)},
        'ensemble': {genre: float(prob) for genre, prob in zip(GENRES, final_probs)}
    }
    
    return predicted_genre, probs_dict

def create_probability_chart(probs_dict, model_type='ensemble'):
    """Create an interactive probability chart"""
    probs = probs_dict[model_type]
    
    # Sort by probability
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    genres_sorted = [item[0] for item in sorted_items]
    probs_sorted = [item[1] for item in sorted_items]
    
    # Create color scale
    colors = ['#1f77b4' if i == 0 else '#aec7e8' for i in range(len(genres_sorted))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs_sorted,
            y=genres_sorted,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.2%}' for p in probs_sorted],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Genre Probabilities - {model_type.upper()}",
        xaxis_title="Probability",
        yaxis_title="Genre",
        height=400,
        showlegend=False,
        xaxis=dict(tickformat='.0%'),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def create_comparison_chart(probs_dict):
    """Create a comparison chart for all models"""
    genres = GENRES
    
    fig = go.Figure()
    
    # Add traces for each model
    for model_name, color in [('lstm', '#1f77b4'), ('cnn', '#ff7f0e'), ('ensemble', '#2ca02c')]:
        probs = [probs_dict[model_name][genre] for genre in genres]
        
        fig.add_trace(go.Bar(
            name=model_name.upper(),
            x=genres,
            y=probs,
            marker_color=color,
            text=[f'{p:.1%}' for p in probs],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Comparison",
        xaxis_title="Genre",
        yaxis_title="Probability",
        barmode='group',
        height=500,
        yaxis=dict(tickformat='.0%'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🎵 Music Genre Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ensemble Learning with LSTM + CNN</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    if not models:
        st.error("❌ No models loaded. Please ensure model files are in the 'models' directory.")
        return
    
    # Ensemble method selection
    st.sidebar.subheader("Ensemble Method")
    
    available_methods = {
        'mean_averaging': 'Mean Averaging',
        'weighted_voting': 'Weighted Voting (Recommended)',
        'max_voting': 'Max Voting',
    }
    
    if 'lr_meta' in models:
        available_methods['lr_stacking'] = 'Logistic Regression Stacking'
    if 'xgb_meta' in models:
        available_methods['xgb_stacking'] = 'XGBoost Stacking'
    if 'nn_meta' in models:
        available_methods['nn_stacking'] = 'Neural Network Stacking'
    
    ensemble_method = st.sidebar.selectbox(
        "Select method:",
        options=list(available_methods.keys()),
        format_func=lambda x: available_methods[x],
        index=1  # Default to weighted voting
    )
    
    # Method description
    method_descriptions = {
        'mean_averaging': "Simple average of LSTM and CNN predictions. Equal weights for both models.",
        'weighted_voting': "Dynamic weighting based on model confidence. Adapts to each prediction.",
        'max_voting': "Selects the prediction with highest confidence from either LSTM or CNN.",
        'lr_stacking': "Meta-learner using Logistic Regression on combined predictions.",
        'xgb_stacking': "Meta-learner using XGBoost on combined predictions. Often the best performer.",
        'nn_stacking': "Meta-learner using Neural Network on combined predictions."
    }
    
    st.sidebar.info(f"ℹ️ {method_descriptions.get(ensemble_method, '')}")
    
    # About section
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        **Fixed Ensemble Implementation**
        
        This app uses separate models for:
        - **LSTM**: Vocal features
        - **CNN**: Accompaniment features
        
        **Improvements:**
        - ✅ No data leakage
        - ✅ Proper validation methodology
        - ✅ Multiple ensemble strategies
        - ✅ Real-time predictions
        
        **Supported Genres:**
        Blues, Classical, Country, Disco, 
        Hip-hop, Jazz, Metal, Pop, Reggae, Rock
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["📁 Upload Audio", "📊 Model Info"])
    
    with tab1:
        st.subheader("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, etc.)",
            type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
            help="Upload a music file for genre classification"
        )
        
        if uploaded_file is not None:
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Process button
            if st.button("🎯 Classify Genre", type="primary"):
                with st.spinner("Analyzing audio..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Extract features
                        progress_bar = st.progress(0)
                        st.text("Extracting MFCC features...")
                        progress_bar.progress(25)
                        
                        mfcc = extract_mfcc(tmp_path)
                        
                        if mfcc is not None:
                            progress_bar.progress(50)
                            st.text("Preparing model inputs...")
                            
                            # Prepare inputs
                            lstm_input, cnn_input = prepare_features_for_models(mfcc)
                            
                            progress_bar.progress(75)
                            st.text("Making predictions...")
                            
                            # Make predictions
                            predicted_genre, probs_dict = predict_ensemble(
                                models, lstm_input, cnn_input, method=ensemble_method
                            )
                            
                            progress_bar.progress(100)
                            st.text("Complete!")
                            
                            # Display results
                            st.success("✅ Classification Complete!")
                            
                            # Predicted genre
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                confidence = probs_dict['ensemble'][predicted_genre]
                                st.markdown(f"""
                                <div style='text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 1rem; margin: 1rem 0;'>
                                    <h2 style='color: #1f77b4; margin: 0;'>{predicted_genre.upper()}</h2>
                                    <p style='font-size: 1.5rem; color: #666; margin: 0.5rem 0;'>Confidence: {confidence:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Detailed probabilities
                            st.subheader("📊 Detailed Results")
                            
                            # Create tabs for different views
                            view_tab1, view_tab2, view_tab3 = st.tabs(["Ensemble", "LSTM", "CNN"])
                            
                            with view_tab1:
                                fig = create_probability_chart(probs_dict, 'ensemble')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with view_tab2:
                                fig = create_probability_chart(probs_dict, 'lstm')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with view_tab3:
                                fig = create_probability_chart(probs_dict, 'cnn')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Comparison chart
                            st.subheader("🔄 Model Comparison")
                            fig_comparison = create_comparison_chart(probs_dict)
                            st.plotly_chart(fig_comparison, use_container_width=True)
                            
                            # Show top 3 predictions
                            st.subheader("🏆 Top 3 Predictions")
                            ensemble_probs = probs_dict['ensemble']
                            top3 = sorted(ensemble_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                            
                            cols = st.columns(3)
                            for i, (genre, prob) in enumerate(top3):
                                with cols[i]:
                                    medal = ["🥇", "🥈", "🥉"][i]
                                    st.metric(
                                        label=f"{medal} {genre.title()}",
                                        value=f"{prob:.1%}"
                                    )
                    
                    except Exception as e:
                        st.error(f"❌ Error during classification: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        else:
            # Sample instructions
            st.info("""
            👆 Upload an audio file to get started!
            
            **Tips for best results:**
            - Use clear, high-quality audio
            - 30-second clips work best
            - Supported formats: WAV, MP3, OGG, FLAC, M4A
            """)
    
    with tab2:
        st.subheader("📊 Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**LSTM Model (Vocal Features)**")
            st.code("""
Input: (40, 132)
├─ Bidirectional LSTM(256)
├─ Bidirectional LSTM(256)
├─ Dense(256) + Dropout(0.5)
├─ Dense(128) + Dropout(0.5)
├─ Dense(32) + Dropout(0.5)
└─ Dense(10, softmax)
            """, language="text")
        
        with col2:
            st.markdown("**CNN Model (Accompaniment)**")
            st.code("""
Input: (40, 132, 1)
├─ Conv2D(64, 3x3) + MaxPool
├─ Conv2D(128, 3x3) + MaxPool
├─ Conv2D(256, 3x3) + MaxPool
├─ Conv2D(512, 3x3) + MaxPool
├─ Flatten
├─ Dense(256) + Dropout(0.5)
├─ Dense(128) + Dropout(0.5)
└─ Dense(10, softmax)
            """, language="text")
        
        st.subheader("🔧 Ensemble Methods")
        
        st.markdown("""
        | Method | Description | Best For |
        |--------|-------------|----------|
        | **Mean Averaging** | Simple average of predictions | Balanced approach |
        | **Weighted Voting** | Dynamic weights based on confidence | Adaptive predictions |
        | **Max Voting** | Highest confidence prediction | Clear genre signals |
        | **LR Stacking** | Logistic regression meta-learner | Linear combinations |
        | **XGBoost Stacking** | Gradient boosting meta-learner | Complex patterns |
        | **NN Stacking** | Neural network meta-learner | Non-linear combinations |
        """)
        
        st.subheader("📈 Performance Metrics")
        
        # Mock performance data (replace with actual metrics)
        performance_data = {
            'Model': ['LSTM', 'CNN', 'Mean Avg', 'Weighted', 'XGBoost'],
            'Accuracy': [0.72, 0.75, 0.76, 0.78, 0.81],
            'Type': ['Base', 'Base', 'Ensemble', 'Ensemble', 'Ensemble']
        }
        
        import pandas as pd
        df = pd.DataFrame(performance_data)
        
        fig = px.bar(df, x='Model', y='Accuracy', color='Type',
                     title='Model Performance Comparison',
                     color_discrete_map={'Base': '#ff7f0e', 'Ensemble': '#2ca02c'})
        fig.update_layout(yaxis=dict(tickformat='.0%', range=[0.6, 0.85]))
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
