import logging
import soundfile as sf
import librosa
from pathlib import Path
import shutil

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def resample_audio(input_path: Path, output_path: Path, target_sr: int) -> bool:
    """Resamples a single WAV audio file to the target sample rate, or copies if already correct.

    Args:
        input_path: Path to the input WAV file.
        output_path: Path to save the resampled WAV file.
        target_sr: The target sample rate in Hz.

    Returns:
        True if resampling/copying was successful, False otherwise.
    """
    try:
        # Get sample rate without loading the whole file first
        info = sf.info(input_path)
        sr = info.samplerate

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if sr == target_sr:
            # Sample rate already matches, just copy the file
            logging.debug(f"Audio already at target sample rate {target_sr}Hz. Copying: {input_path} to {output_path}")
            shutil.copyfile(input_path, output_path)
            return True
        else:
            # Sample rate differs, perform resampling
            logging.debug(f"Resampling required for {input_path} (from {sr}Hz to {target_sr}Hz)")
            # Load as mono for resampling
            y, sr_loaded = librosa.load(input_path, sr=None, mono=True)
            # Double-check loaded SR just in case info was misleading
            if sr_loaded != sr:
                logging.warning(f"Sample rate from sf.info ({sr}) differs from librosa.load ({sr_loaded}) for {input_path}. Using {sr_loaded} for resampling.")
                sr = sr_loaded

            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sf.write(output_path, y_resampled, target_sr, subtype='PCM_16')
            logging.debug(f"Resampled {input_path} (from {sr}Hz) to {output_path} ({target_sr}Hz)")
            return True

    except Exception as e:
        logging.error(f"Error processing file {input_path} for resampling/copying: {e}")
        # Remove potentially incomplete output file
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as rm_e:
                logging.error(f"Failed to remove incomplete output file {output_path}: {rm_e}")
        return False


def process_directory(source_dir: str, target_dir: str, target_sample_rate: int):
    """Processes a directory recursively, resampling WAV files.

    Args:
        source_dir: The directory containing WAV files (output of step 1).
        target_dir: The directory where resampled WAV files will be saved.
        target_sample_rate: The target sample rate in Hz.
    """
    logging.info(f"Starting resampling from '{source_dir}' to '{target_dir}' at {target_sample_rate} Hz")
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    error_count = 0
    skipped_count = 0

    for item in source_path.rglob('*.wav'):
        if item.is_file():
            relative_path = item.relative_to(source_path)
            output_file_path = target_path / relative_path

            # Ensure the output directory for the file exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            if resample_audio(item, output_file_path, target_sample_rate):
                processed_count += 1
            else:
                error_count += 1
        else:
            skipped_count += 1

    logging.info(f"Resampling complete. Processed: {processed_count}, Errors: {error_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    # Example usage: Reads directories and sample rate from config
    process_directory(config.WAV_CONVERSION_DIR,
                      config.RESAMPLED_DIR,
                      config.TARGET_SAMPLE_RATE) 