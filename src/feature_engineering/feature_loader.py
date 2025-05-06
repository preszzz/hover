"""Loads and prepares features from the preprocessed HuggingFace dataset."""

import logging
# import tensorflow as tf # Removed unused import

# Import the feature extractor loading function from our models module
import config
from models.transformer_model import get_feature_extractor
from utils import load_dataset_splits

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Extractor Initialization ---
# Load the feature extractor once when the module is loaded
# This avoids reloading it repeatedly during data processing.
try:
    FEATURE_EXTRACTOR = get_feature_extractor()
    TARGET_SR = FEATURE_EXTRACTOR.sampling_rate
    MAX_LENGTH_SEC = 10 # Example: Max duration in seconds AST expects (often 10s)
    # MAX_LENGTH_SAMPLES = TARGET_SR * MAX_LENGTH_SEC
    logging.info(f"Successfully loaded feature extractor: {FEATURE_EXTRACTOR}")
    logging.info(f"Target sampling rate from feature extractor: {TARGET_SR} Hz")
    logging.info(f"Feature extractor expects max_length: {FEATURE_EXTRACTOR.max_length} samples")
except Exception as e:
    logging.error(f"Failed to load feature extractor: {e}", exc_info=True)
    FEATURE_EXTRACTOR = None
    TARGET_SR = 22050 # Fallback SR if extractor fails, adjust if needed
# --- End Feature Extractor Initialization ---

def preprocess_features(batch):
    """Applies the AST feature extractor to a batch of audio data.

    This function is designed to be used with `dataset.map()`.

    Args:
        batch: A dictionary representing a batch of examples from the HF dataset.
        Expected to contain an 'audio' key with audio data.

    Returns:
        A dictionary containing the processed 'input_features'.
    """
    if FEATURE_EXTRACTOR is None:
        raise RuntimeError("Feature extractor was not loaded successfully. Cannot preprocess.")

    # Ensure audio data is in the expected format (list of numpy arrays)
    audio_arrays = [x["array"] for x in batch['audio']]
    sampling_rate = batch['audio'][0]["sampling_rate"]

    # Check if sampling rate matches the extractor's expected rate
    if sampling_rate != TARGET_SR:
        # This shouldn't happen if load_data_splits cast the column correctly,
        # but include a warning just in case.
        logging.warning(
            f"Audio sampling rate ({sampling_rate} Hz) does not match "
            f"feature extractor target ({TARGET_SR} Hz). Results may be suboptimal."
        )

    # Apply the feature extractor
    # It handles resampling (if needed, though ideally done already),
    # spectrogram computation, normalization, padding, and truncation.
    inputs = FEATURE_EXTRACTOR(
        audio_arrays,
        sampling_rate=sampling_rate, # Use the actual rate from the batch
        return_tensors="np" # Return NumPy arrays suitable for TF
    )

    # The feature extractor typically returns a dictionary.
    # The key for the processed features is often 'input_values' or 'input_features'.
    # For AST, it's usually 'input_values'. Let's rename it for clarity if needed.
    # Check the output keys of your specific extractor if unsure.
    if "input_values" in inputs:
        batch["input_features"] = inputs["input_values"]
    elif "input_features" in inputs:
        batch["input_features"] = inputs["input_features"] # Already named correctly
    else:
        logging.error(f"Feature extractor output did not contain expected keys ('input_values' or 'input_features'). Found: {inputs.keys()}")
        # Handle error appropriately, maybe return None or raise exception
        raise KeyError("Could not find processed features in feature extractor output.")


    # Ensure the label is present and correctly formatted (e.g., integer index)
    # This assumes labels are already numerical indices in the dataset
    if 'label' not in batch:
        raise KeyError(f"Label column 'label' not found in the batch.")
    # batch["labels"] = batch[LABEL_COLUMN] # Keep the label column

    return batch

# Example usage:
if __name__ == "__main__":
    try:
        logging.info("--- Feature Loader Example --- ")
        # 1. Load data
        datasets = load_dataset_splits()
        logging.info(f"Loaded dataset splits: {list(datasets.keys())}")

        # Optional: Inspect the first example of the training set
        if datasets and 'train' in datasets and len(datasets['train']) > 0:
            example = datasets['train'][0]
            logging.info(f"First training example structure: {example.keys()}")
            logging.info(f"Audio sample keys: {example['audio'].keys()}")
            logging.info(f"Audio sampling rate: {example['audio']['sampling_rate']} Hz")
            logging.info(f"Label: {example['label']}")

            # Verify audio column casting worked
            if TARGET_SR and example['audio']['sampling_rate'] != TARGET_SR:
                logging.warning("Mismatch between target SR and example SR after loading!")

        # 2. Apply preprocessing
        if FEATURE_EXTRACTOR:
            logging.info("Applying preprocessing function using .map()...")
            # Note: `batched=True` is crucial for efficiency
            # `num_proc` can be set for parallel processing if needed
            processed_datasets = datasets.map(preprocess_features, batched=True, remove_columns=['audio'])
            logging.info("Preprocessing complete.")

            # Inspect the first processed example
            if processed_datasets and 'train' in processed_datasets and len(processed_datasets['train']) > 0:
                processed_example = processed_datasets['train'][0]
                logging.info(f"First processed training example structure: {processed_example.keys()}")
                logging.info(f"Shape of input_features: {processed_example['input_features'].shape}")
                logging.info(f"Label: {processed_example['label']}")

                # Verify the input shape matches model expectations
                expected_shape = (FEATURE_EXTRACTOR.nb_mel_bins, FEATURE_EXTRACTOR.max_length)
                actual_shape = processed_example['input_features'].shape
                # The extractor might return (batch, bins, time) or (bins, time)
                # Adjust comparison based on map output. map usually keeps batch dim implicit.
                if actual_shape[-2:] == expected_shape:
                    logging.info(f"Processed feature shape {actual_shape} matches expected {expected_shape} (ignoring batch).")
                else:
                    logging.warning(f"Processed feature shape {actual_shape} might not match expected {expected_shape}. Check feature extractor and model input layer.")

        else:
            logging.warning("Skipping preprocessing example as feature extractor failed to load.")

    except Exception as e:
        logging.error(f"Error during feature loader example: {e}", exc_info=True) 