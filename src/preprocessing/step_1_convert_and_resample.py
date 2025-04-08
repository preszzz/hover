import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def convert_and_resample(input_path: Path, output_path: Path, target_sr: int) -> bool:
    """Converts an audio file to WAV format and resamples it in one step if needed.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the converted and resampled WAV file.
        target_sr: Target sampling rate in Hz.

    Returns:
        True if conversion was successful, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Check the original sample rate first
        info = sf.info(input_path)
        original_sr = info.samplerate

        if original_sr == target_sr:
            # If sample rate matches, just convert to WAV without resampling
            logging.debug(f"Converting {input_path} to WAV (no resampling needed, sr={original_sr}Hz)")
            audio_data, _ = librosa.load(input_path, sr=None, mono=False)
        else:
            # Need to convert and resample
            logging.debug(f"Converting and resampling {input_path} from {original_sr}Hz to {target_sr}Hz")
            audio_data, _ = librosa.load(input_path, sr=target_sr, mono=False)

        # Ensure audio_data is 2D (samples, channels) for soundfile write
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]  # Add channel dim
        elif audio_data.ndim > 1:
            # Transpose for soundfile (samples, channels)
            audio_data = audio_data.T

        # Write as WAV file using soundfile
        sf.write(output_path, audio_data, target_sr if original_sr != target_sr else original_sr, subtype='PCM_16')
        logging.debug(f"Processed {input_path} to {output_path}")
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

def process_directory(source_dir: str, target_dir: str, target_sr: int):
    """Processes a directory recursively, converting and resampling audio files to WAV.

    Args:
        source_dir: The root directory containing raw audio files.
        target_dir: The directory where processed WAV files will be saved.
        target_sr: Target sampling rate in Hz.
    """
    logging.info(f"Starting audio conversion and resampling from '{source_dir}' to '{target_dir}'")
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

            # Skip files that aren't supported audio formats
            if not is_supported_audio_file(item):
                logging.info(f"Skipping non-audio file: {item}")
                skipped_non_audio_count += 1
                continue

            # Ensure the output directory for the file exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            if convert_and_resample(item, output_file_path, target_sr):
                processed_count += 1
            else:
                error_count += 1

    logging.info(f"Processing complete. Audio files processed: {processed_count + error_count}, "
                 f"Successfully converted: {processed_count}, Conversion errors: {error_count}, "
                 f"Skipped non-audio files: {skipped_non_audio_count}")

if __name__ == "__main__":
    # Example usage: Reads directories from config
    process_directory(config.RAW_DATA_DIR, config.RESAMPLED_DIR, config.TARGET_SAMPLE_RATE) 