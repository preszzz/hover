"""Hyperparameter tuning script for the audio classification model using Optuna."""

import logging
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

# --- Model Initialization for Trainer HPO ---
def model_init():
    """Initializes a new model for each Optuna trial."""
    logging.debug("Initializing model for HPO trial.")
    # build_transformer_model already handles num_classes and model_checkpoint from config
    model = build_transformer_model(num_classes=2, model_checkpoint=config.MODEL_CHECKPOINT)
    return model.to(DEVICE)

# --- Optuna Hyperparameter Space Definition ---
def optuna_hp_space(trial):
    """Defines the hyperparameter search space for Optuna."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        # Other HPs like dropout_rate, freeze_backbone are omitted for simplicity with Trainer HPO
        # num_train_epochs is fixed in TrainingArguments for HPO trials
    }

# --- Main Hyperparameter Tuning Function ---
def run_hyperparameter_tuning(n_trials: int, study_name: str):
    """Run the hyperparameter tuning process using Hugging Face Trainer.
    
    Args:
        n_trials: Number of Optuna trials to run.
        study_name: Name for the Optuna study.
    """
    # Load raw dataset
    logging.info(f"Initializing dataset")
    try:
        ds = load_dataset_splits(dataset_name=config.DATASET_NAME)
        
        ds = ds.rename_column("audio", "input_values")
        processed_datasets = ds.with_transform(feature_extractor)

        logging.info("Dataset initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize dataset: {e}")
        raise

    logging.info(f"Starting hyperparameter tuning with {n_trials} trials for study '{study_name}'.")

    hpo_output_dir = os.path.join(config.MODEL_SAVE_DIR, "hpo_trainer_output")
    os.makedirs(hpo_output_dir, exist_ok=True)
    logging.info(f"HPO outputs will be saved to {hpo_output_dir}")

    # Base TrainingArguments for HPO
    # Some arguments will be overridden by Optuna during search
    training_args = TrainingArguments(
        output_dir=hpo_output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        num_train_epochs=5,
        logging_steps=10,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        disable_tqdm=False,
    )

    # Initialize Trainer for HPO
    trainer = Trainer(
        model=None,  # Model will be initialized by model_init
        args=training_args,
        model_init=model_init,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["valid"],
        compute_metrics=compute_metrics,
        processing_class=feature_extractor
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
    )

    # Print best trial information
    logging.info("--- Hyperparameter Search Finished ---")
    logging.info("Best trial found:")
    logging.info(f"Run ID (objective value / accuracy): {best_trial_results.objective}")
    logging.info(f"Hyperparameters: {best_trial_results.hyperparameters}")

if __name__ == "__main__":
    N_TRIALS = 20 # Default 20 trials
    STUDY_NAME = "ast_hpo_study_trainer"

    run_hyperparameter_tuning(n_trials=N_TRIALS, study_name=STUDY_NAME) 