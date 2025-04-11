"""Step 2: Process audio files into labeled chunks with features."""

import logging
import numpy as np
from pathlib import Path

# Import config and utilities
import config
from utils import audio_utils, path_utils, label_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_chunk(chunk_data: np.ndarray, output_dir: Path, sr: int) -> bool:
    """Process a single audio chunk: validate, normalize, extract and save features.
    
    Args:
        chunk_data: The audio chunk data as float32 numpy array
        output_dir: Directory to save the features
        sr: Sample rate of the audio data
        
    Returns:
        bool: True if processing was successful, False if validation failed
    """
    try:
        # 1. Validate chunk
        if not audio_utils.validate_chunk(chunk_data, config.CHUNK_LENGTH_SAMPLES):
            return False

        # 2. Normalize audio
        normalized_signal, norm_success = audio_utils.normalize_audio(chunk_data, config.NORMALIZATION_DB)
        if not norm_success:
            return False

        # 3. Extract MFCC
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

        # 4. Save features
        output_dir.mkdir(parents=True, exist_ok=True)
        return audio_utils.process_and_save_features(
            chunk_data,
            output_dir,
            sr,
            config.EXPECTED_FRAMES
        )

    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        if output_dir.exists():
            path_utils.clean_directory(output_dir)
        return False

def process_audio_file(input_path: Path, dataset: str, target_sr: int) -> tuple[int, int]:
    """Process a single audio file: split into chunks and extract features.
    
    Args:
        input_path: Path to the input audio file
        dataset: Name of the dataset
        target_sr: Target sample rate
        
    Returns:
        tuple[int, int]: (chunks_processed, errors)
    """
    chunks_processed = 0
    errors = 0

    try:
        # Get label from path
        label = label_utils.determine_label(input_path)
        if not label:
            logging.error(f"Could not determine label for {input_path}")
            return 0, 1

        # Load and validate audio
        audio_data, sr = audio_utils.load_audio(input_path)
        if sr != target_sr:
            logging.error(f"Unexpected sample rate ({sr} vs {target_sr})")
            return 0, 1

        # Process short file
        if len(audio_data) <= config.CHUNK_LENGTH_SAMPLES:
            if len(audio_data) <= config.CHUNK_LENGTH_SAMPLES // 2:
                logging.info(f"File too short: {input_path}")
                return 0, 1

            # Pad if necessary
            padded_data = np.zeros(config.CHUNK_LENGTH_SAMPLES, dtype=np.float32)
            padded_data[:len(audio_data)] = audio_data
            audio_data = padded_data

            # Process single chunk
            chunk_name = input_path.stem
            chunk_dir = path_utils.get_final_chunk_path(
                Path(config.PROCESSED_DIR),
                dataset,
                label,
                chunk_name
            )
            
            if process_chunk(audio_data, chunk_dir, target_sr):
                chunks_processed += 1
            else:
                errors += 1
            return chunks_processed, errors

        # Process full-length file in chunks
        num_chunks = len(audio_data) // config.CHUNK_LENGTH_SAMPLES
        for i in range(num_chunks):
            start = i * config.CHUNK_LENGTH_SAMPLES
            end = (i + 1) * config.CHUNK_LENGTH_SAMPLES
            chunk_data = audio_data[start:end]

            chunk_name = f"{input_path.stem}_chunk_{i+1}"
            chunk_dir = path_utils.get_final_chunk_path(
                Path(config.PROCESSED_DIR),
                dataset,
                label,
                chunk_name
            )

            if process_chunk(chunk_data, chunk_dir, target_sr):
                chunks_processed += 1
            else:
                errors += 1

    except Exception as e:
        logging.error(f"Error processing file {input_path}: {e}")
        return chunks_processed, errors + 1

    return chunks_processed, errors

def process_directory(source_dir: str, dataset: str, target_sr: int):
    """Process all audio files in the source directory.
    
    Args:
        source_dir: Directory containing resampled audio files
        dataset: Name of the dataset
        target_sr: Expected sample rate of input files
    """
    logging.info(f"Starting processing of dataset '{dataset}' from '{source_dir}'")
    source_path = Path(source_dir)

    total_chunks_processed = 0
    total_errors = 0
    files_processed = 0

    for item in source_path.rglob('*.wav'):
        if item.is_file():
            files_processed += 1
            if files_processed % 100 == 0:
                logging.info(f"Processing file {files_processed}: {item.name}")
            chunks, errors = process_audio_file(item, dataset, target_sr)
            total_chunks_processed += chunks
            total_errors += errors

    logging.info(f"Processing complete. Files processed: {files_processed}, "
                f"Chunks processed: {total_chunks_processed}, Errors: {total_errors}")

if __name__ == "__main__":
    process_directory(config.RESAMPLED_DIR,
                     config.DATASET_NAME,
                     config.TARGET_SAMPLE_RATE)