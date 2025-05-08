# Main entry point for the application
from src.hyperparameter.tune import run_hyperparameter_tuning
from src.feature_engineering.feature_loader import preprocess_features
from src.utils.loader import load_dataset_splits
import config

def run_training():
    ds = load_dataset_splits(dataset_name=config.DATASET_NAME)

    # Preprocess data using Hugging Face `map`
    processed_datasets = ds.map(
        preprocess_features,
        batched=True,
        batch_size=200,
        writer_batch_size=200,
        cache_file_names={
            "train": "train_processed.cache",
            "valid": "valid_processed.cache",
            "test": "test_processed.cache"
        },
        remove_columns=['audio']
    )

    print(processed_datasets)

if __name__ == "__main__":
    run_training()