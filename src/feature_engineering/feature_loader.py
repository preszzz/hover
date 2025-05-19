"""Loads and prepares features from the preprocessed HuggingFace dataset."""

import logging
# import tensorflow as tf # Removed unused import

# Import the feature extractor loading function from our models module
import config
from utils import load_dataset_splits
from models.transformer_model import get_feature_extractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Extractor Initialization ---
# Load the feature extractor once when the module is loaded
# This avoids reloading it repeatedly during data processing.
try:
    feature_extractor = get_feature_extractor(config.MODEL_CHECKPOINT)
    logging.info(f"Successfully loaded feature extractor: {feature_extractor}")
    logging.info(f"Target sampling rate from feature extractor: {feature_extractor.sampling_rate} Hz")
    logging.info(f"Feature extractor expects max_length: {feature_extractor.max_length} samples")
except Exception as e:
    logging.error(f"Failed to load feature extractor: {e}", exc_info=True)
    feature_extractor = None
# --- End Feature Extractor Initialization ---

def preprocess_features(batch):
    """Applies the AST feature extractor to a batch of audio data.

    This function is designed to be used with `dataset.with_transform()`.

    Args:
        batch: A dictionary representing a batch of examples from the HF dataset.
        Expected to contain an 'audio' key with audio data.

    Returns:
        A dictionary containing the processed 'input_values'.
    """
    if feature_extractor is None:
        raise RuntimeError("Feature extractor was not loaded successfully. Cannot preprocess.")

    audio_arrays = [x["array"] for x in batch['input_values']]

    # Apply the feature extractor
    # It handles resampling spectrogram computation, normalization, padding, and truncation.
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=config.CHUNK_LENGTH_SAMPLES,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    if "input_values" in inputs:
        batch["input_values"] = inputs["input_values"]
    else:
        logging.error(f"Feature extractor output did not contain expected keys ('input_values'). Found: {inputs.keys()}")
        raise KeyError("Could not find processed features in feature extractor output.")

    if 'label' not in batch:
        raise KeyError(f"Label column 'label' not found in the batch.")

    return batch

# Example usage:
if __name__ == "__main__":
    try:
        logging.info("--- Feature Loader Example --- ")
        # 1. Load data
        dataset = load_dataset_splits(config.DATASET_NAME)

        # 2. Apply preprocessing
        if feature_extractor:
            dataset = dataset.rename_column('audio', 'input_values')
            processed_datasets = dataset.with_transform(preprocess_features)

            # Inspect the first processed example
            if processed_datasets:
                processed_example = processed_datasets['train'][0]
                logging.info(f"First processed training example structure: {processed_example.keys()}")
                logging.info(f"Shape of input_values: {processed_example['input_values'].shape}")

                # Verify the input shape matches model expectations
                expected_shape = (feature_extractor.num_mel_bins, feature_extractor.max_length)
                actual_shape = processed_example['input_values'].shape
                # The extractor might return (batch, bins, time) or (bins, time)
                # Adjust comparison based on map output. map usually keeps batch dim implicit.
                if actual_shape == expected_shape:
                    logging.info(f"Processed feature shape {actual_shape} matches expected {expected_shape}.")
                else:
                    logging.warning(f"Processed feature shape {actual_shape} might not match expected {expected_shape}. Check feature extractor and model input layer.")

        else:
            logging.warning("Skipping preprocessing example as feature extractor failed to load.")

    except Exception as e:
        logging.error(f"Error during feature loader example: {e}", exc_info=True) 