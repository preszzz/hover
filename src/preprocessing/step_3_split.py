import logging
import numpy as np
import soundfile as sf
from pathlib import Path
import yaml

# Assuming config.py is in the same directory or accessible via PYTHONPATH
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the metadata filename
SOURCE_INFO_FILENAME = "source_info.yaml"

def split_wav_file(input_path: Path, output_base_dir: Path, chunk_length_ms: int, target_sr: int) -> tuple[int, int]:
    """Splits a single WAV file into fixed-length chunks using numpy and soundfile.

    Each chunk is saved into its own subdirectory within a nested structure inside the output_base_dir.
    Example output: output_base_dir/DatasetA/subdir/c/c_chunk_1/c_chunk_1.wav
    Also saves 'source_info.yaml' with metadata in each chunk directory.

    Args:
        input_path: Path to the input WAV file.
        output_base_dir: Base directory for intermediate segment folders.
        chunk_length_ms: Desired chunk length in milliseconds.
        target_sr: Expected sample rate of input file.

    Returns:
        A tuple (chunks_created, errors).
    """
    chunks_created = 0
    errors = 0
    try:
        # Get Original Path Info
        resampled_base_path = Path(config.RESAMPLED_DIR)
        if not input_path.is_relative_to(resampled_base_path):
            logging.error(f"Input path {input_path} is not relative to RESAMPLED_DIR {resampled_base_path}. Cannot determine dataset/relative path correctly. Skipping.")
            return 0, 1

        # Get path relative to the RESAMPLED_DIR (e.g., DatasetA/subdir/c.wav)
        relative_path_from_resampled = input_path.relative_to(resampled_base_path)

        # Get dataset name (first part of the relative path)
        dataset_name = relative_path_from_resampled.parts[0]

        # Get the source path relative within the dataset (e.g., subdir/c.wav)
        relative_source_path_in_dataset = Path(*relative_path_from_resampled.parts[1:])

        # Get path relative to RESAMPLED_DIR without extension for dir structure
        relative_path_no_ext = relative_path_from_resampled.with_suffix('')

        # Get file info first to avoid loading large files entirely if unnecessary
        info = sf.info(input_path)
        samplerate = info.samplerate
        total_frames = info.frames
        channels = info.channels

        if samplerate != target_sr:
            logging.warning(f"Unexpected error: File in RESAMPLED_DIR has incorrect sample rate ({samplerate} vs {target_sr}). Step 2 may have failed. Skipping: {input_path}")
            return 0, 0

        # Use the chunk length from config
        chunk_length_samples = config.CHUNK_LENGTH_SAMPLES

        if chunk_length_samples <= 0:
             logging.error(f"Calculated chunk length in samples is zero or negative ({chunk_length_samples}). Check config. Skipping: {input_path}")
             return 0, 1

        num_full_chunks = total_frames // chunk_length_samples
        
        # Read the audio data
        audio_data, _ = sf.read(input_path, dtype='float32')

        # Determine if we need to process a short file with padding or regular chunks
        if num_full_chunks == 0:
            # Short file case - create one padded chunk
            # file_duration_ms = (total_frames / target_sr) * 1000
            # logging.info(f"File {input_path} shorter than chunk size ({file_duration_ms:.1f}ms < {chunk_length_ms}ms). Padding to chunk length.")
            
            # Create padded version (zero padding at the end)
            padded_data = np.zeros(chunk_length_samples, dtype=np.float32)
            padded_data[:len(audio_data)] = audio_data
            
            # Set up the chunk output path
            chunk_name_base = f"{input_path.stem}"
            chunk_name_wav = f"{chunk_name_base}.wav"
            
            # Define the specific directory structure
            chunk_dir = output_base_dir / relative_path_no_ext / chunk_name_base
            chunk_output_path = chunk_dir / chunk_name_wav
            metadata_output_path = chunk_dir / SOURCE_INFO_FILENAME
            
            try:
                chunk_dir.mkdir(parents=True, exist_ok=True)
                # Write padded chunk with the target sample rate
                sf.write(chunk_output_path, padded_data, target_sr, subtype='PCM_16')
                
                # Write metadata file
                metadata = {
                    'dataset_name': dataset_name,
                    'relative_source_path_in_dataset': relative_source_path_in_dataset.as_posix()
                }
                with open(metadata_output_path, 'w') as f:
                    yaml.safe_dump(metadata, f)
                    
                return 1, 0  # One chunk created, no errors
            except Exception as e:
                logging.error(f"Error writing padded chunk from {input_path} to {chunk_output_path}: {e}")
                return 0, 1
                
        # Process full chunks
        for i in range(num_full_chunks):
            start_sample = i * chunk_length_samples
            end_sample = (i + 1) * chunk_length_samples
            chunk_data = audio_data[start_sample:end_sample]
            
            chunk_index = i + 1  # 1-based index for chunk names
            chunk_name_base = f"{input_path.stem}_chunk_{chunk_index}"
            chunk_name_wav = f"{chunk_name_base}.wav"
            
            # Define paths for this chunk
            chunk_dir = output_base_dir / relative_path_no_ext / chunk_name_base
            # Define the full path for the chunk's WAV file
            chunk_output_path = chunk_dir / chunk_name_wav
            # Define the full path for the metadata file
            metadata_output_path = chunk_dir / SOURCE_INFO_FILENAME
            
            try:
                chunk_dir.mkdir(parents=True, exist_ok=True)
                # Write chunk with the same samplerate and subtype
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

    except Exception as e:
        logging.error(f"Error processing file {input_path} for splitting: {e}")
        return 0, 1  # Indicate error occurred before/during chunk loop

    return chunks_created, errors


def process_directory(source_dir: str, target_dir: str, chunk_length_ms: int, target_sample_rate: int):
    """Processes a directory recursively, splitting WAV files into chunks.

    Args:
        source_dir: The directory containing resampled WAV files (output of step 2).
        target_dir: The base directory where chunk subdirectories will be created.
        chunk_length_ms: The desired length of each chunk in milliseconds.
        target_sample_rate: The expected sample rate of input files.
    """
    logging.info(f"Starting splitting from '{source_dir}' to '{target_dir}' with chunk length {chunk_length_ms} ms")
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)  # Ensure base output dir exists

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
            files_skipped += 1

    logging.info(f"Splitting complete. Files processed: {files_processed}, Chunks created: {total_chunks_created}, Errors: {total_errors}, Skipped items: {files_skipped}")


if __name__ == "__main__":
    # Example usage: Reads directories, chunk length, and SR from config
    process_directory(config.RESAMPLED_DIR,
                      config.INTERIM_SPLIT_DIR,
                      config.CHUNK_LENGTH_MS,
                      config.TARGET_SAMPLE_RATE)