import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import os

def extract_features(audio, sr, feature_type='mfcc', **kwargs):
    """
    Extract audio features from the audio signal.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        feature_type: Type of feature to extract ('mfcc', 'melspectrogram', 'spectral', 'combined')
        **kwargs: Additional parameters for feature extraction
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    if feature_type == 'mfcc' or feature_type == 'combined':
        n_mfcc = kwargs.get('n_mfcc', 20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        features['mfcc_mean'] = mfccs_mean
        features['mfcc_std'] = mfccs_std
    
    if feature_type == 'melspectrogram' or feature_type == 'combined':
        n_mels = kwargs.get('n_mels', 128)
        hop_length = kwargs.get('hop_length', 512)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['melspectrogram'] = mel_spec_db
    
    if feature_type == 'spectral' or feature_type == 'combined':
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
        features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1)
        
        # ZCR - Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
    
    # Root Mean Square Energy
    rms = librosa.feature.rms(y=audio)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return features

def extract_dataset_features(input_dir, output_file, feature_type='combined', **kwargs):
    """
    Extract features from all audio files in the directory and save to CSV.
    
    Args:
        input_dir: Directory containing processed audio files
        output_file: Path to save the features DataFrame
        feature_type: Type of features to extract
        **kwargs: Additional parameters for feature extraction
    """
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                file_paths.append(os.path.join(root, file))
    
    feature_dicts = []
    
    for file_path in tqdm(file_paths, desc="Extracting features"):
        # Determine class label from directory structure
        rel_path = os.path.relpath(file_path, input_dir)
        class_label = os.path.dirname(rel_path).split(os.path.sep)[0]
        
        # Load audio
        audio, sr = librosa.load(file_path)
        
        # Extract features
        features = extract_features(audio, sr, feature_type, **kwargs)
        
        # Add metadata
        features['file_path'] = file_path
        features['class'] = class_label
        
        feature_dicts.append(features)
    
    # Handle non-scalar features
    # This part depends on your specific feature set
    # For now, I'll assume most features are scalars
    
    # Convert to DataFrame
    df = pd.DataFrame(feature_dicts)
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved features for {len(file_paths)} files to {output_file}")
    
    return df

if __name__ == "__main__":
    # Example usage
    extract_dataset_features(
        input_dir="../../data/processed",
        output_file="../../data/features/audio_features.csv",
        feature_type='combined',
        n_mfcc=20,
        n_mels=128
    ) 