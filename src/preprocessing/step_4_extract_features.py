import logging
import shutil
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import os

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_expected_frames(sr: int, duration_ms: int, hop_length: int) -> int:
    """Calculate the expected number of frames for librosa MFCC with center=True."""
    samples = int(sr * duration_ms / 1000)
    # Librosa with center=True pads signal, effectively rounding up frames
    expected_frames = int(np.ceil(float(samples) / hop_length))
    return expected_frames


# Calculate expected dimensions
EXPECTED_FRAMES = calculate_expected_frames(config.TARGET_SAMPLE_RATE,
                                            config.CHUNK_LENGTH_MS,
                                            config.HOP_LENGTH)
# Use the precalculated value from config.py
EXPECTED_SAMPLES = config.CHUNK_LENGTH_SAMPLES


def process_segment_chunk(segment_dir: Path) -> bool:
    """Processes a single audio segment chunk directory.

    Reads the WAV file, validates it, extracts raw signal and MFCC features,
    and saves them. Deletes the directory if validation fails.

    Args:
        segment_dir: Path to the directory containing the chunk .wav file.

    Returns:
        True if processing was successful, False otherwise (and dir deleted).
    """
    wav_files = list(segment_dir.glob('*.wav'))
    if not wav_files:
        logging.warning(f"No WAV file found in {segment_dir}, skipping.")
        return False
    if len(wav_files) > 1:
        logging.warning(f"Multiple WAV files found in {segment_dir}, using first: {wav_files[0]}")

    wav_path = wav_files[0]
    signal_path = segment_dir / config.SIGNAL_FILENAME
    mfcc_path = segment_dir / config.MFCC_FILENAME

    try:
        # 1. Read WAV file
        signal_float32, sr = sf.read(wav_path, dtype='float32')

        # Ensure it's mono
        if signal_float32.ndim > 1:
            logging.warning(f"Audio is not mono ({signal_float32.ndim} channels) in {wav_path}, converting to mono.")
            signal_float32 = librosa.to_mono(signal_float32.T)  # librosa expects shape (channels, samples)

        # 2. Validation (Length)
        if len(signal_float32) != EXPECTED_SAMPLES:
            logging.warning(f"Signal length mismatch ({len(signal_float32)} != {EXPECTED_SAMPLES}) in {wav_path}. Deleting directory.")
            shutil.rmtree(segment_dir)
            return False

        # 3. Validation (Silence)
        # Convert to int16 for saving and silence check
        signal_int16 = (signal_float32 * np.iinfo(np.int16).max).astype(np.int16)
        if np.all(signal_int16 == 0):
            # logging.warning(f"Signal is all zeros in {wav_path}. Deleting directory.")
            shutil.rmtree(segment_dir)
            return False

        # 4. Save Raw Signal (int16)
        np.savetxt(signal_path, signal_int16, delimiter=',', fmt='%d')
        logging.debug(f"Saved signal to {signal_path}")

        # 5. Normalization (using float32 signal)
        max_abs_val = np.max(np.abs(signal_float32))
        if max_abs_val == 0:
            logging.error(f"Max absolute value is zero after passing silence check? Error in {wav_path}. Deleting directory.")
            shutil.rmtree(segment_dir)
            return False

        # Normalize to have peak at -20dB (as per config.NORMALIZATION_DB)
        normalized_signal = signal_float32 / max_abs_val
        normalized_signal *= (10 ** (config.NORMALIZATION_DB / 20))

        # 6. Validation (Normalization result)
        if not np.isfinite(normalized_signal).all():
             logging.warning(f"Non-finite values after normalization in {wav_path}. Deleting directory.")
             shutil.rmtree(segment_dir)
             return False

        # 7. Extract MFCCs
        mfcc = librosa.feature.mfcc(y=normalized_signal,
                                    sr=sr,
                                    n_mfcc=config.N_MFCC,
                                    n_fft=config.N_FFT,
                                    hop_length=config.HOP_LENGTH,
                                    fmax=config.FMAX,
                                    center=True)

        # 8. Validation (MFCC Shape)
        if mfcc.shape != (config.N_MFCC, EXPECTED_FRAMES):
            logging.warning(f"MFCC shape mismatch ({mfcc.shape} != {(config.N_MFCC, EXPECTED_FRAMES)}) in {wav_path}. Deleting directory.")
            shutil.rmtree(segment_dir)
            return False

        # 9. Save MFCCs (npy format)
        np.save(mfcc_path, mfcc)
        logging.debug(f"Saved MFCCs to {mfcc_path}")

        # 10. Clean up intermediate files
        try:
            os.remove(wav_path)
            logging.debug(f"Removed intermediate WAV file: {wav_path}")
        except OSError as e:
            logging.warning(f"Could not remove intermediate WAV file {wav_path}: {e}")

        return True

    except Exception as e:
        logging.error(f"Failed processing segment {segment_dir}: {e}")
        # Clean up directory on any unexpected error during processing
        if segment_dir.exists():
            try:
                shutil.rmtree(segment_dir)
                logging.info(f"Deleted directory due to error: {segment_dir}")
            except OSError as rm_e:
                 logging.error(f"Failed to remove directory {segment_dir} after error: {rm_e}")
        return False


def process_split_directory(split_base_dir: str):
    """Processes all segment chunk directories found within the base split directory.

    Args:
        split_base_dir: The base directory containing the chunk subdirectories.
    """
    logging.info(f"Starting feature extraction for segments in '{split_base_dir}'")
    base_path = Path(split_base_dir)

    # Find all directories - previous pipeline steps ensure these are valid chunk directories
    segment_dirs = [d for d in base_path.rglob('*') if d.is_dir()]
    
    if not segment_dirs:
        logging.warning(f"No segment chunk directories found in {split_base_dir}. Ensure step 3 ran correctly.")
        return

    logging.info(f"Found {len(segment_dirs)} segment directories to process.")

    processed_count = 0
    deleted_count = 0

    for i, segment_dir in enumerate(segment_dirs):
        if (i + 1) % 100 == 0:
             logging.info(f"Processing segment {i+1}/{len(segment_dirs)}: {segment_dir.name}")
        if process_segment_chunk(segment_dir):
            processed_count += 1
        else:
            deleted_count += 1

    logging.info(f"Feature extraction complete. Segments processed successfully: {processed_count}, Segments deleted due to errors/validation: {deleted_count}")


if __name__ == "__main__":
    # Example usage: Reads directory from config
    process_split_directory(config.INTERIM_SPLIT_DIR) 