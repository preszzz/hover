"""Audio processing utilities for the preprocessing pipeline."""

import logging
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Import config from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config

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
    max_abs_val = np.max(np.abs(signal))
    if max_abs_val == 0:
        return signal, False

    # Normalize to target dB
    normalized_signal = signal / max_abs_val
    normalized_signal *= (10 ** (target_db / 20))
    
    # Validate result
    if not np.isfinite(normalized_signal).all():
        return signal, False
        
    return normalized_signal, True

def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc: int, n_fft: int, 
                hop_length: int, fmax: int) -> tuple[np.ndarray, bool]:
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

def save_features(chunk_data: np.ndarray, output_dir: Path, sr: int, expected_frames: int, mfcc: np.ndarray) -> bool:
    """Process audio chunk and save features.
    
    Args:
        chunk_data: Audio chunk data
        output_dir: Output directory for features
        sr: Sample rate
        expected_frames: Expected number of MFCC frames
        
    Returns:
        True if processing successful, False otherwise
    """
    try:
        # 5. Save features
        # Save raw signal as int16 CSV
        signal_int16 = (chunk_data * np.iinfo(np.int16).max).astype(np.int16)
        signal_path = output_dir / config.SIGNAL_FILENAME
        np.savetxt(signal_path, signal_int16, delimiter=',', fmt='%d')
        
        # Save MFCC as NPY
        mfcc_path = output_dir / config.MFCC_FILENAME
        np.save(mfcc_path, mfcc)
        
        return True
        
    except Exception as e:
        logging.error(f"Feature processing failed: {e}")
        return False 