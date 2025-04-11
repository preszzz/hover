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

def process_chunk(chunk_data: np.ndarray, chunk_dir: Path, sr: int) -> bool:
    """Process a single audio chunk: validate, normalize, extract and save features.
    
    Args:
        chunk_data: The audio chunk data as float32 numpy array
        output_dir: Directory to save the features
        sr: Sample rate of the audio data
        
    Returns:
        bool: True if processing was successful, False if validation failed
    """
    try:
        # 1. Validation (Length)
        if len(chunk_data) != EXPECTED_SAMPLES:
            logging.warning(f"Chunk length mismatch ({len(chunk_data)} != {EXPECTED_SAMPLES})")
            return False

        # 2. Convert to int16 for saving and silence check
        signal_int16 = (chunk_data * np.iinfo(np.int16).max).astype(np.int16)
        if np.all(signal_int16 == 0):
            return False  # Silent chunk

        # 3. Save Raw Signal (int16)
        signal_path = chunk_dir / config.SIGNAL_FILENAME
        np.savetxt(signal_path, chunk_data, delimiter=',', fmt='%.6f')

        # 4. Normalization
        max_abs_val = np.max(np.abs(chunk_data))
        if max_abs_val == 0:
            logging.error(f"Max absolute value is zero after passing silence check?")
            return False

        # Normalize to have peak at -20dB
        normalized_signal = chunk_data / max_abs_val
        normalized_signal *= (10 ** (config.NORMALIZATION_DB / 20))

        # 5. Validation (Normalization result)
        if not np.isfinite(normalized_signal).all():
            logging.warning(f"Non-finite values after normalization")
            return False

        # 6. Extract MFCCs
        mfcc = librosa.feature.mfcc(y=normalized_signal,
                                  sr=sr,
                                  n_mfcc=config.N_MFCC,
                                  n_fft=config.N_FFT,
                                  hop_length=config.HOP_LENGTH,
                                  fmax=config.FMAX,
                                  center=True)

        # 7. Validation (MFCC Shape)
        if mfcc.shape != (config.N_MFCC, EXPECTED_FRAMES):
            logging.warning(f"MFCC shape mismatch ({mfcc.shape} != {(config.N_MFCC, EXPECTED_FRAMES)})")
            return False

        # 8. Save MFCCs
        mfcc_path = chunk_dir / config.MFCC_FILENAME
        np.save(mfcc_path, mfcc)

        return True

    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        return False

def process_audio_file(input_path: Path, output_path: Path, target_sr: int, mapping_rules: dict) -> tuple[int, int]:
    """Process a single audio file: split into chunks and extract features.
    
    Args:
        input_path: Path to the input audio file
        output_path: Base directory for output
        target_sr: Target sample rate
        
    Returns:
        tuple[int, int]: (chunks_processed, errors)
    """
    chunks_processed = 0
    errors = 0

    try:
        # Get Original Path Info
        base_path = Path(config.RESAMPLED_DIR)
        if not input_path.is_relative_to(base_path):
            logging.error(f"Input path {input_path} is not relative to RESAMPLED_DIR {base_path}")
            return 0, 1

        # Preserve the exact same directory structure from resampled dir
        relative_path = input_path.relative_to(base_path)
        relative_path_no_ext = relative_path.with_suffix('')

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
            chunk_name_base = input_path.stem
            chunk_dir = path_utils.get_final_chunk_path(output_path, label, chunk_name_base)
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Pad if necessary
            if EXPECTED_SAMPLES / 2 < info.frames < EXPECTED_SAMPLES:
                padded_data = np.zeros(EXPECTED_SAMPLES, dtype=np.float32)
                padded_data[:len(audio_data)] = audio_data
                audio_data = padded_data

            # Process the chunk
            if process_chunk(audio_data, chunk_dir, target_sr):
                chunks_processed += 1
            else:
                if chunk_dir.exists():
                    shutil.rmtree(chunk_dir)
                errors += 1
            return chunks_processed, errors

        # Process full-length file in chunks
        num_full_chunks = info.frames // EXPECTED_SAMPLES
        for i in range(num_full_chunks):
            start_sample = i * EXPECTED_SAMPLES
            end_sample = (i + 1) * EXPECTED_SAMPLES
            chunk_data = audio_data[start_sample:end_sample]

            chunk_index = i + 1
            chunk_name_base = f"{input_path.stem}_chunk_{chunk_index}"
            chunk_dir = output_path / relative_path_no_ext / chunk_name_base
            chunk_dir.mkdir(parents=True, exist_ok=True)

            if process_chunk(chunk_data, chunk_dir, target_sr):
                chunks_processed += 1
            else:
                if chunk_dir.exists():
                    shutil.rmtree(chunk_dir)
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

    total_chunks_processed = 0
    total_errors = 0
    files_processed = 0

    for item in source_path.rglob('*.wav'):
        if item.is_file():
            files_processed += 1
            if files_processed % 100 == 0:
                logging.info(f"Processing file {files_processed}: {item.name}")
            chunks, errors = process_audio_file(item, target_path, target_sr, mapping_rules)
            total_chunks_processed += chunks
            total_errors += errors

    logging.info(f"Processing complete. Files processed: {files_processed}, "
                f"Chunks processed: {total_chunks_processed}, Errors: {total_errors}")

if __name__ == "__main__":
    process_directory(config.RESAMPLED_DIR,
                      config.PROCESSED_DATA_DIR,
                      config.TARGET_SAMPLE_RATE)