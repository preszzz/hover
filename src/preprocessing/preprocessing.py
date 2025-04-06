import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def load_audio_file(file_path, sr=44100):
    """
    Load an audio file with the specified sample rate.
    
    Args:
        file_path: Path to the audio file
        sr: Sample rate to load the audio file with
        
    Returns:
        audio: Audio time series
        sr: Sample rate
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def preprocess_audio(audio, sr, target_sr=22050, normalize=True):
    """
    Preprocess audio data.
    
    Args:
        audio: Audio time series
        sr: Original sample rate
        target_sr: Target sample rate
        normalize: Whether to normalize the audio
        
    Returns:
        Preprocessed audio data
    """
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Normalize if requested
    if normalize:
        audio = librosa.util.normalize(audio)
    
    return audio

def process_dataset(input_dir, output_dir, sr=22050, normalize=True):
    """
    Process all audio files in the given directory.
    
    Args:
        input_dir: Directory containing raw audio files
        output_dir: Directory to save processed files
        sr: Target sample rate
        normalize: Whether to normalize audio
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                file_paths.append(os.path.join(root, file))
    
    for file_path in tqdm(file_paths, desc="Processing audio files"):
        # Extract relative path to maintain directory structure
        rel_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load and process
        audio, orig_sr = load_audio_file(file_path)
        if audio is not None:
            processed_audio = preprocess_audio(audio, orig_sr, sr, normalize)
            
            # Save processed file
            sf.write(output_path, processed_audio, sr)
            
    print(f"Processed {len(file_paths)} audio files")

if __name__ == "__main__":
    # Example usage
    process_dataset(
        input_dir="../../data/raw",
        output_dir="../../data/processed",
        sr=22050,
        normalize=True
    ) 