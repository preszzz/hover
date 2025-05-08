# Main entry point for the application
from src.training.train import train_model
from src.hyperparameter.tune import run_hyperparameter_tuning


if __name__ == "__main__":
    run_hyperparameter_tuning(n_trials=20, study_name="ast_hyperparameter_tuning")