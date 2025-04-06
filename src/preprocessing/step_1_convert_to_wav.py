import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_to_wav(input_path: Path, output_path: Path) -> bool:
    """Converts a single audio file (MP3, WAV, etc. supported by librosa/soundfile) to WAV format.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the converted WAV file.

    Returns:
        True if conversion was successful or file was already WAV, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Attempt to load the audio file using librosa
        # Librosa will use soundfile or audioread (potentially with ffmpeg)
        # We load with sr=None to preserve original sample rate for now
        audio_data, samplerate = librosa.load(input_path, sr=None, mono=False)

        # Ensure audio_data is 2D (samples, channels) for soundfile write
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis] # Add channel dim
        elif audio_data.ndim > 1:
            # Librosa loads as (channels, samples), transpose for soundfile (samples, channels)
             audio_data = audio_data.T

        # Write the audio data as a WAV file using soundfile
        # Specify subtype for consistency (e.g., PCM_16)
        sf.write(output_path, audio_data, samplerate, subtype='PCM_16')
        logging.debug(f"Converted/Copied {input_path} (sr={samplerate}) to {output_path}")
        return True

    except Exception as e:
        # Catch potential errors during loading (e.g., unsupported format, corrupt file)
        # or writing
        logging.error(f"Error processing file {input_path}: {e}")
        # Attempt to remove potentially incomplete output file
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
    logging.info(f"Starting WAV conversion (using librosa/soundfile) from '{source_dir}' to '{target_dir}'")
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_non_audio_count = 0
    error_count = 0

    # supported_extensions = librosa.util.list_audio_formats() # This was incorrect
    # logging.info(f"Librosa reports supported extensions (via soundfile/audioread): {supported_extensions}")
    # Instead of pre-filtering, we rely on the try-except block in convert_to_wav

    for item in source_path.rglob('*'): # rglob includes subdirectories
        if item.is_file():
            # file_ext = item.suffix.lower()[1:] # No longer needed
            # # Check if librosa *might* support the extension
            # # Note: Actual support depends on backend (soundfile/audioread+ffmpeg)
            # if file_ext in supported_extensions: # This check is removed

            # Directly attempt conversion for any file found.
            # convert_to_wav will handle errors for unsupported/corrupt files.
            relative_path = item.relative_to(source_path)
            output_file_path = target_path / relative_path.with_suffix('.wav')

            # Ensure the output directory for the file exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            if convert_to_wav(item, output_file_path):
                processed_count += 1
            else:
                error_count += 1
                # Log the skipped file here if desired, though convert_to_wav already logs errors
                skipped_non_audio_count += 1 # Increment skipped count on error
            # else:
            #      # This block is removed as we now attempt all files
            #      logging.debug(f"Skipping file with potentially unsupported extension '{file_ext}': {item}")
            #      skipped_non_audio_count += 1

    logging.info(f"WAV conversion complete. Files attempted: {processed_count + error_count}. Successful: {processed_count}, Errors/Skipped: {error_count}")

if __name__ == "__main__":
    # Example usage: Reads directories from config
    # Ensure RAW_DATA_DIR contains the audio datasets you want to process
    # This script expects subfolders within RAW_DATA_DIR if your data is organized that way
    process_directory(config.RAW_DATA_DIR, config.WAV_CONVERSION_DIR) 