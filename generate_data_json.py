"""
Generate data.json from audio files in Data/genres_original/

This script processes all audio files in the Data/genres_original/ directory,
extracts MFCC features, and saves them to Data/data.json

The generated file will contain:
- mfcc: MFCC features for each audio segment (shape: samples x 130 x 13)
- labels: Genre labels for each segment
- mapping: Dictionary mapping genre names to numeric labels
"""

import numpy as np
import librosa
import json
import os
import math
from pathlib import Path
from tqdm import tqdm

# Parameters for MFCC extraction
SAMPLE_RATE = 22050      # Sampling rate in Hz
DURATION = 30            # Track duration in seconds
N_FFT = 2048            # FFT window size
HOP_LENGTH = 512        # Number of samples between frames
N_MFCC = 40             # Number of MFCC coefficients (use 40 for better resolution)
NUM_SEGMENTS = 10       # Split each track into 10 segments (3 sec each)

# Paths
DATA_DIR = Path("Data")
AUDIO_DIR = DATA_DIR / "genres_original"
OUTPUT_FILE = DATA_DIR / "data.json"

def get_mfccs_from_audio(audio_dir, sr=SAMPLE_RATE, duration=DURATION, 
                         n_fft=N_FFT, hop_length=HOP_LENGTH, 
                         n_mfcc=N_MFCC, num_segments=NUM_SEGMENTS):
    """
    Process audio files and extract MFCC features
    
    Args:
        audio_dir: Directory containing genre subdirectories with audio files
        sr: Sampling rate
        duration: Duration of audio to process
        n_fft: FFT window size
        hop_length: Number of samples between frames
        n_mfcc: Number of MFCC coefficients
        num_segments: Number of segments to split each track into
    
    Returns:
        dict: Dictionary with 'mfcc', 'labels', and 'mapping' keys
    """
    data = {
        "mfcc": [],
        "labels": [],
        "mapping": []
    }
    
    # Calculate samples per segment
    samples_per_track = sr * duration
    samples_per_segment = int(samples_per_track / num_segments)
    mfccs_per_segment = math.ceil(samples_per_segment / hop_length)
    
    # Get all genre directories
    genre_dirs = sorted([d for d in audio_dir.iterdir() if d.is_dir()])
    
    if not genre_dirs:
        raise ValueError(f"No genre directories found in {audio_dir}")
    
    # Create genre mapping
    data["mapping"] = [d.name for d in genre_dirs]
    
    print(f"\n{'='*60}")
    print(f"MFCC Feature Extraction Started")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Sample Rate: {sr} Hz")
    print(f"  - Duration: {duration} seconds")
    print(f"  - N_MFCC: {n_mfcc} coefficients")
    print(f"  - Segments: {num_segments} per track")
    print(f"  - Expected MFCC shape per segment: ({mfccs_per_segment}, {n_mfcc})")
    print(f"  - Genres found: {len(genre_dirs)}")
    print(f"{'='*60}\n")
    
    # Process each genre
    for genre_idx, genre_dir in enumerate(genre_dirs):
        genre_name = genre_dir.name
        audio_files = list(genre_dir.glob("*.wav")) + list(genre_dir.glob("*.au"))
        
        if not audio_files:
            print(f"⚠️  No audio files found in {genre_name}")
            continue
        
        print(f"Processing {genre_name.upper()}: {len(audio_files)} files")
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc=f"  {genre_name}", leave=False):
            try:
                # Load audio
                audio, _ = librosa.load(str(audio_file), sr=sr, duration=duration)
                
                # Process each segment
                for seg_idx in range(num_segments):
                    # Calculate segment boundaries
                    start_sample = seg_idx * samples_per_segment
                    end_sample = start_sample + samples_per_segment
                    
                    # Extract MFCC for this segment
                    mfcc = librosa.feature.mfcc(
                        y=audio[start_sample:end_sample],
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mfcc=n_mfcc
                    )
                    
                    # Transpose to (time_steps, features)
                    mfcc = mfcc.T
                    
                    # Only append if we have the correct number of time steps
                    if len(mfcc) == mfccs_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(genre_name)
                    
            except Exception as e:
                print(f"    ⚠️  Error processing {audio_file.name}: {str(e)}")
                continue
        
        print(f"  ✓ Completed {genre_name}: {data['labels'].count(genre_name)} segments collected\n")
    
    return data

def save_json(data, output_path):
    """Save data to JSON file"""
    print(f"{'='*60}")
    print("Saving data to disk...")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"✓ Saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"{'='*60}\n")

def main():
    """Main function to generate data.json"""
    
    # Check if audio directory exists
    if not AUDIO_DIR.exists():
        print(f"❌ Error: Audio directory not found: {AUDIO_DIR}")
        print(f"Please make sure audio files are in: {AUDIO_DIR.absolute()}")
        return
    
    # Count audio files
    total_files = sum(1 for genre_dir in AUDIO_DIR.iterdir() 
                     if genre_dir.is_dir() 
                     for _ in list(genre_dir.glob("*.wav")) + list(genre_dir.glob("*.au")))
    
    if total_files == 0:
        print(f"❌ Error: No audio files found in {AUDIO_DIR}")
        return
    
    print(f"\n🎵 Music Genre Classification - Data Generation")
    print(f"Found {total_files} audio files to process\n")
    
    # Generate MFCC features
    data = get_mfccs_from_audio(AUDIO_DIR)
    
    # Convert to numpy for shape checking
    mfcc_array = np.array(data["mfcc"])
    labels_array = np.array(data["labels"])
    
    # Print summary
    print(f"\n{'='*60}")
    print("Data Collection Summary")
    print(f"{'='*60}")
    print(f"Total segments: {len(data['labels'])}")
    print(f"MFCC shape: {mfcc_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"\nGenre distribution:")
    for genre in data["mapping"]:
        count = data["labels"].count(genre)
        print(f"  {genre:12s}: {count:4d} segments")
    print(f"{'='*60}\n")
    
    # Save to JSON
    save_json(data, OUTPUT_FILE)
    
    print("✅ Data generation complete!")
    print(f"\nNext steps:")
    print(f"1. Generate accompaniment_mfcc.json (if needed)")
    print(f"2. Run: python retrain_base_models.py")
    print(f"3. Run: python save_meta_models.py")
    print(f"4. Test: .\\run_streamlit.bat")

if __name__ == "__main__":
    main()
