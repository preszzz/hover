import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_to_wav(input_path: Path, output_path: Path) -> bool:
    """Converts a single audio file to WAV format.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the converted WAV file.

    Returns:
        True if conversion was successful, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load the audio file using librosa with original sample rate
        audio_data, samplerate = librosa.load(input_path, sr=None, mono=False)

        # Ensure audio_data is 2D (samples, channels) for soundfile write
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]  # Add channel dim
        elif audio_data.ndim > 1:
            # Transpose for soundfile (samples, channels)
            audio_data = audio_data.T

        # Write as WAV file using soundfile
        sf.write(output_path, audio_data, samplerate, subtype='PCM_16')
        logging.debug(f"Converted/Copied {input_path} (sr={samplerate}) to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing file {input_path}: {e}")
        # Remove incomplete output file if it exists
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as rm_e:
                logging.error(f"Failed to remove incomplete output file {output_path}: {rm_e}")
        return False


def process_directory(source_dir: str, target_dir: str):
    """Processes a directory recursively, converting supported audio files to WAV.

    Args:
        source_dir: The root directory containing raw audio files.
        target_dir: The directory where WAV files will be saved, preserving structure.
    """
    logging.info(f"Starting WAV conversion from '{source_dir}' to '{target_dir}'")
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_non_audio_count = 0
    error_count = 0

    for item in source_path.rglob('*'):  # rglob includes subdirectories
        if item.is_file():
            relative_path = item.relative_to(source_path)
            output_file_path = target_path / relative_path.with_suffix('.wav')

            # Ensure the output directory for the file exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            if convert_to_wav(item, output_file_path):
                processed_count += 1
            else:
                error_count += 1
                skipped_non_audio_count += 1

    logging.info(f"WAV conversion complete. Files attempted: {processed_count + error_count}. Successful: {processed_count}, Errors/Skipped: {error_count}")

if __name__ == "__main__":
    # Example usage: Reads directories from config
    process_directory(config.RAW_DATA_DIR, config.WAV_CONVERSION_DIR) 