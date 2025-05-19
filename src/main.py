# Main entry point for the application
# from hyperparameter.tune import run_hyperparameter_tuning
from feature_engineering.feature_loader import preprocess_features
from utils.loader import load_dataset_splits
from utils.hardware import list_audio_devices
from deploy.live_detection import main
# from training.train import train_model
import config

def run_preprocessing():
    ds = load_dataset_splits(dataset_name=config.DATASET_NAME)

    # Preprocess data using Hugging Face `map`
    processed_datasets = ds.map(
        preprocess_features,
        batched=True,
        batch_size=200,
        writer_batch_size=200,
        cache_file_names={
            "train": 'cache/train_processed',
            "valid": 'cache/valid_processed',
            "test": 'cache/test_processed'
        },
        remove_columns=['audio']
    )

    print(processed_datasets)

if __name__ == "__main__":
    # train_model()
    # list_audio_devices()
    main()