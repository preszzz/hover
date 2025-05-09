#!/usr/bin/env python
"""Hyperparameter tuning script for the audio classification model using Optuna."""

import logging
import optuna
import os
from transformers import Trainer, TrainingArguments

# Import from project modules
import config
from src.utils.loader import load_dataset_splits
from src.feature_engineering.feature_loader import feature_extractor
from src.models.transformer_model import build_transformer_model
from src.training.train import get_device, compute_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Setup ---
DEVICE = get_device()
logging.info(f"Device set to: {DEVICE}")

# Load raw dataset
logging.info(f"Loading raw dataset: {config.DATASET_NAME}")
try:
    # ds will be used directly by the Trainer with feature_extractor as processing_class
    ds = load_dataset_splits(dataset_name=config.DATASET_NAME)
    logging.info("Raw dataset loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load raw dataset: {e}")
    raise

# --- Model Initialization for Trainer HPO ---
def model_init():
    """Initializes a new model for each Optuna trial."""
    logging.debug("Initializing model for HPO trial.")
    # build_transformer_model already handles num_classes and model_checkpoint from config
    model = build_transformer_model(num_classes=2, model_checkpoint=config.MODEL_CHECKPOINT)
    return model.to(DEVICE) # Ensure model is on the correct device from the start

# --- Optuna Hyperparameter Space Definition ---
def optuna_hp_space(trial: optuna.trial.Trial) -> dict:
    """Defines the hyperparameter search space for Optuna."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        # Other HPs like dropout_rate, freeze_backbone are omitted for simplicity with Trainer HPO
        # num_train_epochs is fixed in TrainingArguments for HPO trials
    }

# --- Main Hyperparameter Tuning Function ---
def run_hyperparameter_tuning(n_trials: int, study_name: str, hpo_epochs: int = 5):
    """Run the hyperparameter tuning process using Hugging Face Trainer.
    
    Args:
        n_trials: Number of Optuna trials to run.
        study_name: Name for the Optuna study.
        hpo_epochs: Number of epochs to train each HPO trial.
    """
    logging.info(f"Starting hyperparameter tuning with {n_trials} trials for study '{study_name}'. Each trial runs for {hpo_epochs} epochs.")

    hpo_output_dir = os.path.join(config.MODEL_SAVE_DIR, "hpo_trainer_output")
    os.makedirs(hpo_output_dir, exist_ok=True)
    logging.info(f"HPO outputs will be saved to {hpo_output_dir}")

    # Base TrainingArguments for HPO
    # Some arguments will be overridden by Optuna during search
    training_args = TrainingArguments(
        output_dir=hpo_output_dir, # Directory for trial outputs
        eval_strategy="epoch",    # Evaluate at the end of each epoch
        save_strategy="no",       # Don't save checkpoints during HPO, only find best params
        num_train_epochs=hpo_epochs,
        logging_steps=10,         # Log more frequently during HPO
        report_to="none",         # Disable reporting to W&B/Tensorboard for HPO by default
        load_best_model_at_end=False, # Not needed as we only care about best HPs
        metric_for_best_model="accuracy", # Metric to optimize (from compute_metrics)
        disable_tqdm=True,        # Disable progress bars for cleaner HPO logs
        # Ensure other necessary args are set, e.g., from config or fixed for HPO
        # learning_rate, per_device_train_batch_size, weight_decay will be set by Optuna
    )

    # Initialize Trainer for HPO
    trainer = Trainer(
        model=None,  # Model will be initialized by model_init
        args=training_args,
        model_init=model_init,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        compute_metrics=compute_metrics,
        processing_class=feature_extractor # Use feature_extractor to process raw data
        # data_collator is not needed when processing_class is used with default collator
    )

    # Run hyperparameter search
    logging.info("Starting Optuna hyperparameter search with Trainer...")
    best_trial_results = trainer.hyperparameter_search(
        direction="maximize",       # We want to maximize accuracy
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=n_trials,
        study_name=study_name,
        # compute_objective: If None, Trainer uses metric_for_best_model from compute_metrics
        # pruner: Can specify an Optuna pruner here, e.g., optuna.pruners.MedianPruner()
        # storage: Can specify Optuna storage URL, e.g., "sqlite:///db.sqlite3"
    )

    # Print best trial information
    logging.info("--- Hyperparameter Search Finished ---")
    logging.info("Best trial found:")
    logging.info(f"  Run ID (objective value / accuracy): {best_trial_results.objective}")
    logging.info(f"  Hyperparameters: {best_trial_results.hyperparameters}")

    # Clean up HPO output directory if desired (optional)
    # import shutil
    # shutil.rmtree(hpo_output_dir)
    # logging.info(f"Cleaned up HPO output directory: {hpo_output_dir}")

if __name__ == "__main__":
    N_TRIALS = config.HPO_N_TRIALS if hasattr(config, 'HPO_N_TRIALS') else 20 # Default 20 trials
    STUDY_NAME = config.HPO_STUDY_NAME if hasattr(config, 'HPO_STUDY_NAME') else "ast_hpo_study_trainer"
    HPO_EPOCHS = config.HPO_EPOCHS if hasattr(config, 'HPO_EPOCHS') else 5 # Default 5 epochs per trial

    # Ensure feature_extractor is loaded before starting
    if feature_extractor is None:
        logging.error("Feature extractor not loaded. Exiting HPO.")
    else:
        logging.info("Feature extractor loaded successfully.")
        run_hyperparameter_tuning(n_trials=N_TRIALS, study_name=STUDY_NAME, hpo_epochs=HPO_EPOCHS) 