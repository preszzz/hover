"""Main script to run the audio preprocessing pipeline."""

import logging
import time

# Import config
import config
import step_1_resample
import step_2_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    """Executes the audio preprocessing pipeline:
    1. Convert and resample audio files
    2. Process audio (extract features and create labels)
    """
    start_time = time.time()
    logging.info("--- Starting Audio Preprocessing Pipeline ---")

    # Step 1: Convert and resample audio files in one step
    logging.info("--- Running Step 1: Convert and Resample Audio ---")
    step_1_start = time.time()
    try:
        step_1_resample.process_directory(
            config.RAW_DATA_DIR,
            config.INTERIM_DATA_DIR,
            config.TARGET_SAMPLE_RATE
        )
    except Exception as e:
        logging.error(f"Error during Step 1 (Convert and Resample): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 1.")
        return
    step_1_duration = time.time() - step_1_start
    logging.info(f"--- Step 1 Finished (Duration: {step_1_duration:.2f} seconds) ---")

    # Step 2: Process audio and create labels
    logging.info("--- Running Step 2: Process Audio and Create Labels ---")
    step_2_start = time.time()
    try:
        step_2_process.process_directory(
            config.INTERIM_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.TARGET_SAMPLE_RATE
        )
    except Exception as e:
        logging.error(f"Error during Step 2 (Split and Extract): {e}", exc_info=True)
        logging.critical("Pipeline halted due to error in Step 2.")
        return
    step_2_duration = time.time() - step_2_start
    logging.info(f"--- Step 2 Finished (Duration: {step_2_duration:.2f} seconds) ---")

    # Pipeline summary
    total_duration = time.time() - start_time
    logging.info("--- Audio Preprocessing Pipeline Complete ---")
    logging.info(f"Total Duration: {total_duration:.2f}s")
    logging.info(f"Processed Data: {config.PROCESSED_DATA_DIR}")
    logging.info(f"Interim Data: {config.INTERIM_DATA_DIR}")

if __name__ == "__main__":
    run_pipeline() 