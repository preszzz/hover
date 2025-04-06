import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_features_from_audio(audio, sr, feature_config=None):
    """
    Extract multiple acoustic features from audio data.
    
    Args:
        audio: Audio time series data
        sr: Sample rate
        feature_config: Dictionary with feature extraction configuration
    
    Returns:
        Dictionary containing extracted features
    """
    if feature_config is None:
        feature_config = {
            'mfcc': {'n_mfcc': 20, 'include': True},
            'spectral': {'include': True},
            'temporal': {'include': True},
            'chroma': {'include': False},
            'mel': {'include': False, 'n_mels': 128},
        }
    
    features = {}
    
    # Make sure audio isn't empty or invalid
    if audio is None or len(audio) == 0:
        print("Warning: Empty or invalid audio")
        return features
    
    # MFCC features - capture timbral characteristics
    if feature_config['mfcc']['include']:
        n_mfcc = feature_config['mfcc'].get('n_mfcc', 20)
        
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            
            # Get statistics for each coefficient
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_vars = np.var(mfccs, axis=1)
            mfcc_max = np.max(mfccs, axis=1)
            mfcc_min = np.min(mfccs, axis=1)
            
            # Add to feature dictionary
            for i in range(n_mfcc):
                features[f'mfcc_mean_{i+1}'] = mfcc_means[i]
                features[f'mfcc_var_{i+1}'] = mfcc_vars[i]
                features[f'mfcc_max_{i+1}'] = mfcc_max[i]
                features[f'mfcc_min_{i+1}'] = mfcc_min[i]
                
            # Add delta and delta-delta features (derivatives)
            if feature_config['mfcc'].get('deltas', True):
                mfcc_delta = librosa.feature.delta(mfccs)
                mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
                
                # Get statistics for delta coefficients
                delta_means = np.mean(mfcc_delta, axis=1)
                delta_vars = np.var(mfcc_delta, axis=1)
                
                # Get statistics for delta2 coefficients
                delta2_means = np.mean(mfcc_delta2, axis=1)
                delta2_vars = np.var(mfcc_delta2, axis=1)
                
                # Add delta features
                for i in range(n_mfcc):
                    features[f'mfcc_delta_mean_{i+1}'] = delta_means[i]
                    features[f'mfcc_delta_var_{i+1}'] = delta_vars[i]
                    features[f'mfcc_delta2_mean_{i+1}'] = delta2_means[i]
                    features[f'mfcc_delta2_var_{i+1}'] = delta2_vars[i]
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
    
    # Spectral features - capture frequency characteristics
    if feature_config['spectral']['include']:
        try:
            # Spectral centroid - center of mass of the spectrum
            spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spec_centroid)
            features['spectral_centroid_var'] = np.var(spec_centroid)
            
            # Spectral bandwidth - weighted standard deviation around spectral centroid
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spec_bandwidth)
            features['spectral_bandwidth_var'] = np.var(spec_bandwidth)
            
            # Spectral contrast - difference between peaks and valleys in the spectrum
            # Particularly useful for distinguishing different types of sounds
            spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral_contrast_mean'] = np.mean(np.mean(spec_contrast, axis=1))
            features['spectral_contrast_var'] = np.mean(np.var(spec_contrast, axis=1))
            
            # Spectral rolloff - frequency below which a specified percentage of the spectrum is contained
            spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
            features['spectral_rolloff_var'] = np.var(spec_rolloff)
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
    
    # Temporal features - capture time-domain characteristics
    if feature_config['temporal']['include']:
        try:
            # Zero crossing rate - rate at which the signal changes sign
            # Useful for distinguishing voiced/unvoiced sounds
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_var'] = np.var(zcr)
            
            # Root mean square energy - volume/energy of the signal
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_var'] = np.var(rms)
            
            # Temporal flatness - measure of how noise-like vs. tone-like the signal is
            # Using the spectral flatness as an approximation
            flatness = librosa.feature.spectral_flatness(y=audio)[0]
            features['flatness_mean'] = np.mean(flatness)
            features['flatness_var'] = np.var(flatness)
        except Exception as e:
            print(f"Error extracting temporal features: {e}")
    
    # Chroma features - relate to the 12 pitch classes in music
    # Less relevant for drone detection but could be useful
    if feature_config['chroma']['include']:
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma_mean'] = np.mean(np.mean(chroma, axis=1))
            features['chroma_var'] = np.mean(np.var(chroma, axis=1))
        except Exception as e:
            print(f"Error extracting chroma features: {e}")
    
    # Mel spectrogram features - represent the frequency content using mel scale
    # Particularly useful for drone detection as it models human hearing
    if feature_config['mel']['include']:
        try:
            n_mels = feature_config['mel'].get('n_mels', 128)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Get statistics for each band
            mel_means = np.mean(mel_spec_db, axis=1)
            mel_vars = np.var(mel_spec_db, axis=1)
            
            # Add selected bands to features (using fewer bands to keep feature count manageable)
            # Focus on bands that capture drone motor frequencies
            selected_bands = np.linspace(0, n_mels-1, 20, dtype=int)
            for i, band in enumerate(selected_bands):
                features[f'mel_mean_{i+1}'] = mel_means[band]
                features[f'mel_var_{i+1}'] = mel_vars[band]
        except Exception as e:
            print(f"Error extracting mel spectrogram features: {e}")
    
    return features

def segment_audio(audio, sr, segment_duration=1.0, overlap=0.5):
    """
    Segment audio into smaller chunks with overlap.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments (0.0 to 1.0)
        
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    
    # Calculate number of segments
    num_segments = max(1, int((len(audio) - segment_samples) / hop_samples) + 1)
    
    segments = []
    for i in range(num_segments):
        start = i * hop_samples
        end = start + segment_samples
        
        # Handle last segment
        if end > len(audio):
            # Pad with zeros if needed
            segment = np.zeros(segment_samples)
            segment[:len(audio) - start] = audio[start:]
        else:
            segment = audio[start:end]
        
        segments.append(segment)
    
    return segments

def process_audio_to_features(audio_path, class_label=None, segment_duration=1.0, overlap=0.5, feature_config=None):
    """
    Process a single audio file into features with segmentation.
    
    Args:
        audio_path: Path to the audio file
        class_label: Label for the audio file (e.g., drone type)
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments
        feature_config: Configuration for feature extraction
        
    Returns:
        List of feature dictionaries, one per segment
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Skip if audio is too short
        if len(audio) / sr < 0.5:  # Skip very short files
            print(f"Warning: {audio_path} is too short, skipping")
            return []
        
        # Segment audio
        segments = segment_audio(audio, sr, segment_duration, overlap)
        
        # Extract features for each segment
        all_features = []
        for i, segment in enumerate(segments):
            features = extract_features_from_audio(segment, sr, feature_config)
            
            # Add metadata
            if features:
                features['file_path'] = audio_path
                features['segment_id'] = i
                if class_label is not None:
                    features['class'] = class_label
                
                all_features.append(features)
        
        return all_features
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []

def process_dataset_to_csv(input_dir, output_csv, segment_duration=1.0, overlap=0.5, feature_config=None, 
                           batch_size=100, max_files=None):
    """
    Process all audio files in a directory to a CSV file with features.
    Uses batching to handle large datasets.
    
    Args:
        input_dir: Directory containing audio files organized in subdirectories by class
        output_csv: Path to save the CSV file
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments
        feature_config: Configuration for feature extraction
        batch_size: Number of files to process before writing to CSV
        max_files: Maximum number of files to process (None for all)
    """
    # Get all audio files with their classes
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                # Get class from directory name
                rel_path = os.path.relpath(root, input_dir)
                class_name = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path
                
                # Skip if class_name is empty (files in the root directory)
                if class_name:
                    audio_files.append((os.path.join(root, file), class_name))
    
    # Limit the number of files if specified
    if max_files is not None and max_files < len(audio_files):
        audio_files = audio_files[:max_files]
    
    # Process files in batches
    all_features = []
    file_count = 0
    first_batch = True
    
    for batch_idx in range(0, len(audio_files), batch_size):
        batch_files = audio_files[batch_idx:batch_idx + batch_size]
        batch_features = []
        
        # Process each file in the batch
        for file_path, class_name in tqdm(batch_files, desc=f"Processing batch {batch_idx//batch_size + 1}/{(len(audio_files)-1)//batch_size + 1}"):
            file_features = process_audio_to_features(
                file_path, class_name, segment_duration, overlap, feature_config)
            batch_features.extend(file_features)
            file_count += 1
        
        # Convert batch to DataFrame
        if batch_features:
            df_batch = pd.DataFrame(batch_features)
            
            # Write to CSV
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            
            if first_batch:
                df_batch.to_csv(output_csv, index=False, mode='w')
                first_batch = False
            else:
                # Append without header
                df_batch.to_csv(output_csv, index=False, mode='a', header=False)
        
        print(f"Processed {file_count}/{len(audio_files)} files, extracted {len(batch_features)} segments in this batch")
    
    print(f"Complete! Processed {file_count} files to {output_csv}")
    return output_csv

if __name__ == "__main__":
    # Example configuration
    feature_config = {
        'mfcc': {'n_mfcc': 20, 'include': True, 'deltas': True},
        'spectral': {'include': True},
        'temporal': {'include': True},
        'chroma': {'include': False},
        'mel': {'include': True, 'n_mels': 128},
    }
    
    # Process dataset
    process_dataset_to_csv(
        input_dir='../../data/raw',
        output_csv='../../data/processed/features_dataset.csv',
        segment_duration=1.0,
        overlap=0.5,
        feature_config=feature_config,
        batch_size=50
    ) 