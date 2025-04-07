import logging
import time
from pathlib import Path

# Assuming config.py and step scripts are in the same directory or accessible via PYTHONPATH
import config
import step_1_convert_to_wav
import step_2_resample
import step_3_split
import step_4_extract_features
import step_5_create_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_pipeline():
    """Executes the full audio preprocessing pipeline step-by-step."""

    start_time = time.time()
    logging.info("--- Starting Audio Preprocessing Pipeline ---")

    # Step 1: Convert all audio in RAW_DATA_DIR to WAV format in WAV_CONVERSION_DIR
    logging.info("--- Running Step 1: Convert to WAV ---")
    step_1_start = time.time()
    try:
        step_1_convert_to_wav.process_directory(config.RAW_DATA_DIR,
                                                config.WAV_CONVERSION_DIR)
    except Exception as e:
        logging.error(f"Error during Step 1 (Convert to WAV): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 1.")
        return
    step_1_duration = time.time() - step_1_start
    logging.info(f"--- Step 1 Finished (Duration: {step_1_duration:.2f} seconds) ---")

    # Step 2: Resample WAV files to TARGET_SAMPLE_RATE in RESAMPLED_DIR
    logging.info("--- Running Step 2: Resample Audio ---")
    step_2_start = time.time()
    try:
        step_2_resample.process_directory(config.WAV_CONVERSION_DIR,
                                        config.RESAMPLED_DIR,
                                        config.TARGET_SAMPLE_RATE)
    except Exception as e:
        logging.error(f"Error during Step 2 (Resample Audio): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 2.")
        return
    step_2_duration = time.time() - step_2_start
    logging.info(f"--- Step 2 Finished (Duration: {step_2_duration:.2f} seconds) ---")

    # Step 3: Split resampled audio into chunks in INTERIM_SPLIT_DIR
    logging.info("--- Running Step 3: Split Audio into Chunks ---")
    step_3_start = time.time()
    try:
        step_3_split.process_directory(config.RESAMPLED_DIR,
                                     config.INTERIM_SPLIT_DIR,
                                     config.CHUNK_LENGTH_MS,
                                     config.TARGET_SAMPLE_RATE)
    except Exception as e:
        logging.error(f"Error during Step 3 (Split Audio): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 3.")
        return
    step_3_duration = time.time() - step_3_start
    logging.info(f"--- Step 3 Finished (Duration: {step_3_duration:.2f} seconds) ---")

    # Step 4: Extract features (signal.csv, mfcc.npy) for each chunk in INTERIM_SPLIT_DIR
    logging.info("--- Running Step 4: Extract Features ---")
    step_4_start = time.time()
    try:
        step_4_extract_features.process_split_directory(config.INTERIM_SPLIT_DIR)
    except Exception as e:
        logging.error(f"Error during Step 4 (Extract Features): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 4.")
        return
    step_4_duration = time.time() - step_4_start
    logging.info(f"--- Step 4 Finished (Duration: {step_4_duration:.2f} seconds) ---")

    # Step 5: Create Label Files (reading from INTERIM_SPLIT_DIR)
    logging.info("--- Running Step 5: Create Label Files & Move to Processed ---")
    step_5_start = time.time()
    try:
        script_dir = Path(__file__).parent
        mapping_file = script_dir / 'label_mapping.yaml'
        step_5_create_label.create_label_files(config.INTERIM_SPLIT_DIR,
                                               config.WAV_CONVERSION_DIR,
                                               str(mapping_file))
    except Exception as e:
        logging.error(f"Error during Step 5 (Create Label Files): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 5.")
        return
    step_5_duration = time.time() - step_5_start
    logging.info(f"--- Step 5 Finished (Duration: {step_5_duration:.2f} seconds) ---")

    total_duration = time.time() - start_time
    logging.info(f"--- Audio Preprocessing Pipeline Finished Successfully --- ")
    logging.info(f"Total execution time: {total_duration:.2f} seconds")
    logging.info(f"Processed data (features and labels) located in: {config.PROCESSED_DATA_DIR}")
    logging.info(f"Intermediate files located in: {config.INTERIM_DATA_DIR}")
    logging.info(f"Labeling rules defined in: src/preprocessing/label_mapping.yaml")


if __name__ == "__main__":
    run_pipeline() 