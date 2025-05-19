"""Hyperparameter tuning script for the audio classification model using Optuna."""

import logging
import os
from transformers import Trainer, TrainingArguments

# Import from project modules
import config
from feature_engineering.feature_loader import feature_extractor
from models.transformer_model import build_transformer_model
from utils.loader import load_dataset_splits
from utils.metric import compute_metrics
from utils.hardware import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Setup ---
DEVICE = get_device()
logging.info(f"Device set to: {DEVICE}")

# --- Model Initialization for Trainer HPO ---
def model_init():
    """Initializes a new model for each Optuna trial."""
    logging.debug("Initializing model for HPO trial.")
    model = build_transformer_model(num_classes=2, model_checkpoint=config.MODEL_CHECKPOINT)
    return model.to(DEVICE)

# --- Optuna Hyperparameter Space Definition ---
def optuna_hp_space(trial):
    """Defines the hyperparameter search space for Optuna."""
    return {
        "hub_model_id": f"preszzz/{config.STUDY_NAME}-trial-{trial.number}",
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "polynomial"]),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 1.0),
        "optim": trial.suggest_categorical("optim", ["adamw_torch", "adafactor", "adamw_torch_fused"])
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
        model=None,
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
    )

    # Print best trial information
    logging.info("--- Hyperparameter Search Finished ---")
    logging.info("Best trial found:")
    logging.info(f"Run ID (objective value / accuracy): {best_trial_results.objective}")
    logging.info(f"Hyperparameters: {best_trial_results.hyperparameters}")

if __name__ == "__main__":
    N_TRIALS = 20
    STUDY_NAME = "ast_hpo_study_trainer"

    run_hyperparameter_tuning(n_trials=N_TRIALS, study_name=STUDY_NAME) 