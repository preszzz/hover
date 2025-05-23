"""Audio processing utilities for the preprocessing pipeline."""

import logging
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def is_supported_audio_file(file_path: Path) -> bool:
    """Check if the file has a supported audio format based on its extension.
    
    Args:
        file_path: Path to the audio file to check.
        
    Returns:
        bool: True if the file extension is in the list of supported formats.
    """
    supported_formats = sf.available_formats()
    file_ext = file_path.suffix.upper().lstrip('.') if file_path.suffix else ""
    return file_ext in supported_formats

def validate_chunk(chunk_data: np.ndarray, expected_samples: int) -> bool:
    """Validate audio chunk before processing.
    
    Args:
        chunk_data: Audio chunk data
        expected_samples: Expected number of samples
        
    Returns:
        True if chunk is valid, False otherwise
    """
    # Length validation
    if len(chunk_data) != expected_samples:
        logging.warning(f"Chunk length mismatch ({len(chunk_data)} != {expected_samples})")
        return False
        
    # Convert to int16 for silence check
    signal_int16 = (chunk_data * np.iinfo(np.int16).max).astype(np.int16)
    if np.all(signal_int16 == 0):
        return False  # Silent chunk
        
    return True

def normalize_audio(signal: np.ndarray, target_db: float) -> tuple[np.ndarray, bool]:
    """Normalize audio signal to target dB level.
    
    Args:
        signal: Audio signal
        target_db: Target peak level in dB
        
    Returns:
        Tuple of (normalized signal, success flag)
    """
    # Handle silent/empty signals
    if np.max(np.abs(signal)) < 1e-10:
        return signal, False
        
    try:
        # Calculate current RMS and dB
        rms = np.sqrt(np.mean(signal ** 2))
        current_db = 20 * np.log10(rms)
        
        # Calculate gain needed
        gain_db = target_db - current_db
        gain_factor = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized_signal = signal * gain_factor
        
        # Prevent clipping if needed
        if np.max(np.abs(normalized_signal)) > 0.95:  # Using 0.95 as a safety margin
            normalized_signal = 0.95 * normalized_signal / np.max(np.abs(normalized_signal))
        
        # Validate result
        if not np.isfinite(normalized_signal).all():
            return signal, False
            
        return normalized_signal, True
        
    except Exception as e:
        logging.warning(f"Normalization failed: {e}")
        return signal, False

def extract_mfcc(
        signal: np.ndarray,
        sr: int,
        n_mfcc: int,
        n_fft: int,
        hop_length: int,
        fmax: int
) -> tuple[np.ndarray, bool]:
    """Extract MFCC features from audio signal.
    
    Args:
        signal: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        fmax: Maximum frequency
        
    Returns:
        Tuple of (MFCC features, success flag)
    """
    try:
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            fmax=fmax,
            center=True
        )
        return mfcc, True
    except Exception as e:
        logging.error(f"MFCC extraction failed: {e}")
        return None, False

def save_features(
        chunk_data: np.ndarray,
        mfcc: np.ndarray,
        label: str,
        output_dir: Path,
        signal_filename: str,
        mfcc_filename: str,
        label_filename: str
) -> bool:
    """Save audio features and label to files.
    
    Args:
        chunk_data: Raw audio chunk data
        mfcc: MFCC features array
        output_dir: Directory to save features
        signal_filename: Filename for raw signal NPY
        mfcc_filename: Filename for MFCC NPY
        label: label string to write to label file
        
    Returns:
        True if save successful, False otherwise
    """
    try:
        # Save raw signal as NPY
        signal_path = output_dir / signal_filename
        np.save(signal_path, chunk_data)
        
        # Save MFCC as NPY
        mfcc_path = output_dir / mfcc_filename
        np.save(mfcc_path, mfcc)
        
        # Save label as txt
        label_path = output_dir / label_filename
        with open(label_path, 'w') as f:
            f.write(label)
        
        return True
        
    except Exception as e:
        logging.error(f"Feature processing failed: {e}")
        return False