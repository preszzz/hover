"""Step 2: Process audio files into labeled chunks with features."""

import logging
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Import config and utilities
import config
from utils import audio_utils, path_utils, label_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Calculate expected dimensions once
EXPECTED_SAMPLES = config.CHUNK_LENGTH_SAMPLES
EXPECTED_FRAMES = int(np.ceil(float(EXPECTED_SAMPLES) / config.HOP_LENGTH))

def process_chunk(chunk_data: np.ndarray, output_dir: Path, sr: int, label: str) -> bool:
    """Process a single audio chunk: validate, normalize, extract and save features.
    
    Args:
        chunk_data: The audio chunk data as float32 numpy array
        output_dir: Directory to save the features
        sr: Sample rate of the audio data
        label: Label string to write to label file
        
    Returns:
        bool: True if processing was successful, False if validation failed
    """
    try:
        # Validation (Length)
        if not audio_utils.validate_chunk(chunk_data, EXPECTED_SAMPLES):
            return False

        # Normalize audio
        normalized_signal, norm_success = audio_utils.normalize_audio(chunk_data, config.NORMALIZATION_DB)
        if not norm_success:
            return False
        
        # Extract MFCC
        mfcc, mfcc_success = audio_utils.extract_mfcc(
            normalized_signal,
            sr,
            config.N_MFCC,
            config.N_FFT,
            config.HOP_LENGTH,
            config.FMAX
        )
        if not mfcc_success:
            return False

        # Validation (MFCC Shape)
        if mfcc.shape != (config.N_MFCC, EXPECTED_FRAMES):
            logging.warning(f"MFCC shape mismatch ({mfcc.shape} != {(config.N_MFCC, EXPECTED_FRAMES)})")
            return False

        # Save features and label
        return audio_utils.save_features(
            chunk_data,
            mfcc,
            label,
            output_dir,
            config.SIGNAL_FILENAME,
            config.MFCC_FILENAME,
            config.LABEL_FILENAME
        )

    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        if output_dir.exists():
            path_utils.clean_directory(output_dir)
        return False

def process_audio_file(input_path: Path, output_path: Path, target_sr: int, mapping_rules: dict) -> tuple[int, int]:
    """Process a single audio file: split into chunks and extract features.
    
    Args:
        input_path: Path to the input audio file
        output_path: Base directory for output
        target_sr: Target sample rate
        mapping_rules: Dictionary of labeling rules
        
    Returns:
        tuple[int, int]: (chunks_processed, errors)
    """
    chunks_processed = 0
    errors = 0

    try:
        # Get Original Path Info
        base_path = Path(config.INTERIM_DATA_DIR)
        if not input_path.is_relative_to(base_path):
            logging.error(f"Input path {input_path} is not relative to INTERIM_DATA_DIR {base_path}")
            return 0, 1

        # Preserve the exact same directory structure from resampled dir
        relative_path = input_path.relative_to(base_path)
        dataset_name = relative_path.parts[0]

        # Get label from path
        label = label_utils.get_label(relative_path, mapping_rules)
        if not label:
            logging.error(f"Could not determine label for {input_path}")
            return 0, 1

        # Verify sample rate
        info = sf.info(input_path)
        if info.samplerate != target_sr:
            logging.warning(f"Unexpected sample rate ({info.samplerate} vs {target_sr})")
            return 0, 1

        # Read the audio data
        audio_data, _ = sf.read(input_path, dtype='float32')

        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data.T)

        # Handle short file
        if info.frames <= EXPECTED_SAMPLES:
            if info.frames <= EXPECTED_SAMPLES // 2:
                logging.info(f"File too short: {input_path}")
                return 0, 1

            # Pad if necessary
            padded_data = np.zeros(EXPECTED_SAMPLES, dtype=np.float32)
            padded_data[:len(audio_data)] = audio_data
            audio_data = padded_data

            chunk_name = input_path.stem
            chunk_dir = path_utils.get_final_chunk_path(output_path, dataset_name, label, chunk_name)
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Process the chunk
            if process_chunk(audio_data, chunk_dir, target_sr, label):
                chunks_processed += 1
            else:
                errors += 1
            return chunks_processed, errors

        # Process full-length file in chunks
        num_full_chunks = info.frames // EXPECTED_SAMPLES
        for i in range(num_full_chunks):
            start = i * EXPECTED_SAMPLES
            end = (i + 1) * EXPECTED_SAMPLES
            chunk_data = audio_data[start:end]

            chunk_name = f"{input_path.stem}_chunk_{i + 1}"
            chunk_dir = path_utils.get_final_chunk_path(output_path, dataset_name, label, chunk_name)
            chunk_dir.mkdir(parents=True, exist_ok=True)

            if process_chunk(chunk_data, chunk_dir, target_sr, label):
                chunks_processed += 1
            else:
                errors += 1

    except Exception as e:
        logging.error(f"Error processing file {input_path}: {e}")
        return chunks_processed, errors + 1

    return chunks_processed, errors

def process_directory(source_dir: str, target_dir: str, target_sr: int):
    """Process all audio files in the source directory.
    
    Args:
        source_dir: Directory containing resampled audio files
        target_dir: Directory to save processed chunks with features
        target_sr: Expected sample rate of input files
    """
    script_dir = Path(__file__).parent
    mapping_file = script_dir / 'label_mapping.yaml'
    mapping_rules = label_utils.load_label_mapping(mapping_file)
    if mapping_rules is None:
        logging.critical("Failed to load label mapping. Aborting label creation.")
        return
    
    logging.info(f"Starting split and feature extraction from '{source_dir}' to '{target_dir}'")
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Count total files first
    total_files = sum(1 for _ in source_path.rglob('*.wav') if _.is_file())
    logging.info(f"Found {total_files} WAV files to process")

    total_chunks_processed = 0
    total_errors = 0
    files_processed = 0

    for item in source_path.rglob('*.wav'):
        if item.is_file():
            files_processed += 1
            if files_processed % 100 == 0:
                progress_percent = (files_processed / total_files) * 100
                logging.info(f"Progress: {files_processed} of {total_files} files ({progress_percent:.1f}%)")
            chunks, errors = process_audio_file(item, target_path, target_sr, mapping_rules)
            total_chunks_processed += chunks
            total_errors += errors

    logging.info(f"Processing complete. Files processed: {files_processed}, "
                f"Chunks processed: {total_chunks_processed}, Errors: {total_errors}")

if __name__ == "__main__":
    process_directory(config.INTERIM_DATA_DIR,
                      config.PROCESSED_DATA_DIR,
                      config.TARGET_SAMPLE_RATE)