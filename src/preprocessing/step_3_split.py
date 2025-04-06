import logging
import numpy as np
import soundfile as sf
from pathlib import Path
import yaml # Added for metadata output

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the metadata filename
SOURCE_INFO_FILENAME = "source_info.yaml"

def split_wav_file(input_path: Path, output_base_dir: Path, chunk_length_ms: int, target_sr: int) -> tuple[int, int]:
    """Splits a single WAV file into fixed-length chunks using numpy and soundfile.

    Each chunk is saved into its own subdirectory within a *nested* structure
    inside the output_base_dir, preserving the original relative path.
    Example output structure for input '.../interim/resampled/DatasetA/subdir/c.wav' and chunk 1:
    output_base_dir / 'DatasetA/subdir/c' / 'c_chunk_1' / 'c_chunk_1.wav'
    Also saves 'source_info.yaml' in the chunk directory.

    Args:
        input_path: Path to the input WAV file (assumed to be at target_sr).
                     Example: /path/to/project/data/interim/resampled/DatasetA/subdir/c.wav
        output_base_dir: The base directory where intermediate segment folders will be created.
                         Example: /path/to/project/data/interim/split_with_meta
        chunk_length_ms: The desired length of each chunk in milliseconds.
        target_sr: The sample rate the input file is expected to have.

    Returns:
        A tuple (chunks_created, errors).
    """
    chunks_created = 0
    errors = 0
    try:
        # --- Get Original Path Info ---
        # Ensure RESAMPLED_DIR is the correct base for calculating relative paths
        resampled_base_path = Path(config.RESAMPLED_DIR)
        if not input_path.is_relative_to(resampled_base_path):
            logging.error(f"Input path {input_path} is not relative to RESAMPLED_DIR {resampled_base_path}. Cannot determine dataset/relative path correctly. Skipping.")
            return 0, 1

        # Get path relative to the RESAMPLED_DIR (e.g., DatasetA/subdir/c.wav)
        relative_path_from_resampled = input_path.relative_to(resampled_base_path)

        # Get dataset name (first part of the relative path)
        dataset_name = relative_path_from_resampled.parts[0]

        # Get the source path relative *within* the dataset (e.g., subdir/c.wav)
        # This is what label mapping rules expect
        relative_source_path_in_dataset = Path(*relative_path_from_resampled.parts[1:])

        # Get path relative to RESAMPLED_DIR *without extension* for dir structure
        relative_path_no_ext = relative_path_from_resampled.with_suffix('')

        # Get file info first to avoid loading large files entirely if unnecessary
        info = sf.info(input_path)
        samplerate = info.samplerate
        total_frames = info.frames
        channels = info.channels

        if samplerate != target_sr:
            logging.warning(f"File sample rate ({samplerate}) does not match target ({target_sr}). Skipping: {input_path}")
            return 0, 0 # Not an error, but prevents incorrect splitting

        chunk_length_samples = int(target_sr * chunk_length_ms / 1000)

        if chunk_length_samples <= 0:
             logging.error(f"Calculated chunk length in samples is zero or negative ({chunk_length_samples}). Check config. Skipping: {input_path}")
             return 0, 1

        num_full_chunks = total_frames // chunk_length_samples

        if num_full_chunks == 0:
            logging.warning(f"File is shorter than chunk length ({chunk_length_ms}ms / {chunk_length_samples} samples), skipping: {input_path}")
            return 0, 0 # Not an error, just no chunks created

        # Read the whole file data (consider block processing for very large files)
        # If memory is a concern, use sf.blocks or SoundFile context manager with read
        audio_data, _ = sf.read(input_path, dtype='float32') # Read as float for consistency

        for i in range(num_full_chunks):
            start_sample = i * chunk_length_samples
            end_sample = (i + 1) * chunk_length_samples
            chunk_data = audio_data[start_sample:end_sample]

            chunk_index = i + 1 # 1-based index for chunk names
            chunk_name_base = f"{input_path.stem}_chunk_{chunk_index}"
            chunk_name_wav = f"{chunk_name_base}.wav"

            # Define the specific directory for this chunk (NESTED STRUCTURE)
            # output_base_dir / DatasetName/subdir/filename_stem / chunk_name_base
            chunk_dir = output_base_dir / relative_path_no_ext / chunk_name_base
            # Define the full path for the chunk's WAV file
            chunk_output_path = chunk_dir / chunk_name_wav
            # Define the full path for the metadata file
            metadata_output_path = chunk_dir / SOURCE_INFO_FILENAME

            try:
                chunk_dir.mkdir(parents=True, exist_ok=True)
                # Write chunk with the same samplerate and subtype as step 2 output
                sf.write(chunk_output_path, chunk_data, target_sr, subtype='PCM_16')
                logging.debug(f"Exported chunk {chunk_index}/{num_full_chunks} to {chunk_output_path}")

                # Write metadata file
                metadata = {
                    'dataset_name': dataset_name,
                    # Use as_posix() for cross-platform path consistency in YAML
                    'relative_source_path_in_dataset': relative_source_path_in_dataset.as_posix()
                }
                with open(metadata_output_path, 'w') as f:
                    yaml.safe_dump(metadata, f)
                logging.debug(f"Wrote metadata to {metadata_output_path}")

                chunks_created += 1
            except Exception as e:
                logging.error(f"Error writing chunk {chunk_index} from {input_path} to {chunk_output_path}: {e}")
                errors += 1
                if chunk_dir.exists():
                    try:
                        # shutil.rmtree(chunk_dir) # Consider implications
                        pass
                    except OSError as rm_e:
                         logging.error(f"Failed to remove potentially incomplete chunk directory {chunk_dir}: {rm_e}")

    except Exception as e:
        logging.error(f"Error processing file {input_path} for splitting: {e}")
        return 0, 1 # Indicate error occurred before/during chunk loop

    # logging.info(f"Finished splitting {input_path}: {chunks_created} chunks created, {errors} errors.")
    return chunks_created, errors


def process_directory(source_dir: str, target_dir: str, chunk_length_ms: int, target_sample_rate: int):
    """Processes a directory recursively, splitting WAV files into chunks.

    Args:
        source_dir: The directory containing resampled WAV files (output of step 2).
        target_dir: The base directory where chunk subdirectories will be created.
        chunk_length_ms: The desired length of each chunk in milliseconds.
        target_sample_rate: The expected sample rate of input files.
    """
    logging.info(f"Starting splitting (soundfile/numpy) from '{source_dir}' to '{target_dir}' with chunk length {chunk_length_ms} ms")
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True) # Ensure base output dir exists

    total_chunks_created = 0
    total_errors = 0
    files_processed = 0
    files_skipped = 0

    for item in source_path.rglob('*.wav'):
        if item.is_file():
            files_processed += 1
            chunks, errors = split_wav_file(item, target_path, chunk_length_ms, target_sample_rate)
            total_chunks_created += chunks
            total_errors += errors
        else:
            files_skipped += 1 # Should not happen with rglob('*.wav')

    logging.info(f"Splitting complete. Files processed: {files_processed}, Chunks created: {total_chunks_created}, Errors: {total_errors}, Skipped items: {files_skipped}")

if __name__ == "__main__":
    # Example usage: Reads directories, chunk length, and SR from config
    # Assumes step_2 has already run and populated RESAMPLED_DIR
    process_directory(config.RESAMPLED_DIR,
                      config.INTERIM_SPLIT_DIR, # Use new intermediate dir
                      config.CHUNK_LENGTH_MS,
                      config.TARGET_SAMPLE_RATE) # Pass target SR