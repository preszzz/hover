"""Loads and prepares features from the preprocessed HuggingFace dataset."""

import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (Consider moving to a central config file) ---
HUGGINGFACE_DATASET_ID = "hover-d/test" # Replace with your actual dataset ID
CACHE_DIR = "./hf_cache" # Optional: specify a cache directory

def load_data(split: str = 'train', dataset_id: str = HUGGINGFACE_DATASET_ID):
    """Loads a specific split (train, valid, test) from the HuggingFace dataset.

    Args:
        split: The dataset split to load ('train', 'valid', or 'test').
        dataset_id: The HuggingFace dataset identifier.

    Returns:
        A HuggingFace Dataset object for the specified split.
    """
    logging.info(f"Loading '{split}' split from dataset: {dataset_id}")
    try:
        dataset = load_dataset(dataset_id, split=split, cache_dir=CACHE_DIR)
        logging.info(f"Successfully loaded '{split}' split with {len(dataset)} examples.")
        # Example: Print features of the loaded dataset
        # logging.info(f"Dataset features: {dataset.features}")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset '{dataset_id}', split '{split}': {e}", exc_info=True)
        raise

def get_feature_loaders(dataset_id: str = HUGGINGFACE_DATASET_ID):
    """Loads the train, validation, and test datasets.

    Args:
        dataset_id: The HuggingFace dataset identifier.

    Returns:
        A tuple containing the train, validation, and test Dataset objects.
    """
    train_dataset = load_data('train', dataset_id)
    valid_dataset = load_data('valid', dataset_id)
    test_dataset = load_data('test', dataset_id)
    return train_dataset, valid_dataset, test_dataset

def preprocess_features(batch):
    """Prepares a batch of data for model training.

    This function might involve:
    - Selecting specific features (e.g., MFCCs, deltas).
    - Combining features (e.g., stacking MFCC, delta, delta2).
    - Reshaping data for the model input (e.g., adding channel dimension for CNN).
    - Potentially converting data types (e.g., to TensorFlow tensors).

    Args:
        batch: A dictionary representing a batch of data from the Dataset.

    Returns:
        A dictionary with the processed features and labels suitable for the model.
    """
    # Example: Select MFCCs and stack them (adjust shape based on model needs)
    # mfcc = batch['mfcc']
    # delta = batch['delta']
    # delta2 = batch['delta2']
    # combined_features = np.stack([mfcc, delta, delta2], axis=-1) # Example stacking

    # Placeholder: Return the batch as is for now
    # Modify this function based on your chosen model's input requirements
    
    # Ensure label is returned
    # Example: return {"features": combined_features, "label": batch["label"]}
    pass # Replace with actual preprocessing logic


if __name__ == "__main__":
    # Example usage:
    logging.info("Testing feature loader...")
    try:
        train_ds, valid_ds, test_ds = get_feature_loaders()
        logging.info(f"Train dataset size: {len(train_ds)}")
        logging.info(f"Validation dataset size: {len(valid_ds)}")
        logging.info(f"Test dataset size: {len(test_ds)}")

        # Example: Accessing the first training example
        if len(train_ds) > 0:
            first_example = train_ds[0]
            logging.info(f"First training example keys: {first_example.keys()}")
            logging.info(f"First training example label: {first_example['label']}")
            # logging.info(f"First training example MFCC shape: {np.array(first_example['mfcc']).shape}")

        # Example: Applying preprocessing (demonstration)
        # train_ds_processed = train_ds.map(preprocess_features, batched=True, batch_size=100)
        # logging.info("Preprocessing function applied (example). Check the function for details.")

    except Exception as e:
        logging.error(f"Error during feature loader test: {e}", exc_info=True) 