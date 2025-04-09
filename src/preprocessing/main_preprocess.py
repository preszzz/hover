import logging
import time
from pathlib import Path

# Assuming config.py and step scripts are in the same directory or accessible via PYTHONPATH
import config
import step_1_resample
import step_2_process
import step_5_create_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_pipeline():
    """Executes the full audio preprocessing pipeline step-by-step."""

    start_time = time.time()
    logging.info("--- Starting Audio Preprocessing Pipeline ---")

    # Step 1: Convert and resample audio files in one step
    logging.info("--- Running Step 1: Convert and Resample Audio ---")
    step_1_start = time.time()
    try:
        step_1_resample.process_directory(
            config.RAW_DATA_DIR,
            config.RESAMPLED_DIR,
            config.TARGET_SAMPLE_RATE
        )
    except Exception as e:
        logging.error(f"Error during Step 1 (Convert and Resample): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 1.")
        return
    step_1_duration = time.time() - step_1_start
    logging.info(f"--- Step 1 Finished (Duration: {step_1_duration:.2f} seconds) ---")

    # Step 2: Split audio and extract features
    logging.info("--- Running Step 2: Split Audio and Extract Features ---")
    step_2_start = time.time()
    try:
        step_2_process.process_directory(
            config.RESAMPLED_DIR,
            config.INTERIM_SPLIT_DIR,
            config.TARGET_SAMPLE_RATE
        )
    except Exception as e:
        logging.error(f"Error during Step 2 (Split and Extract): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 2.")
        return
    step_2_duration = time.time() - step_2_start
    logging.info(f"--- Step 2 Finished (Duration: {step_2_duration:.2f} seconds) ---")

    # Step 3: Create Label Files (reading from INTERIM_SPLIT_DIR)
    logging.info("--- Running Step 3: Create Label Files & Move to Processed ---")
    step_3_start = time.time()
    try:
        script_dir = Path(__file__).parent
        mapping_file = script_dir / 'label_mapping.yaml'
        step_5_create_label.create_label_files(config.INTERIM_SPLIT_DIR,
                                             str(mapping_file))
    except Exception as e:
        logging.error(f"Error during Step 3 (Create Label Files): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 3.")
        return
    step_3_duration = time.time() - step_3_start
    logging.info(f"--- Step 3 Finished (Duration: {step_3_duration:.2f} seconds) ---")

    total_duration = time.time() - start_time
    logging.info(f"--- Audio Preprocessing Pipeline Finished Successfully --- ")
    logging.info(f"Total execution time: {total_duration:.2f} seconds")
    logging.info(f"Processed data (features and labels) located in: {config.PROCESSED_DATA_DIR}")
    logging.info(f"Intermediate files located in: {config.INTERIM_DATA_DIR}")
    logging.info(f"Labeling rules defined in: src/preprocessing/label_mapping.yaml")


if __name__ == "__main__":
    run_pipeline() 