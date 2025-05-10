"""Main training script for the audio classification model using PyTorch."""

import logging
import os
from transformers import Trainer, TrainingArguments

# Import from project modules
import config
from feature_engineering.feature_loader import preprocess_features, feature_extractor
from models.transformer_model import build_transformer_model
from utils import load_dataset_splits
from utils.metric import compute_metrics
from utils.hardware import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = get_device()

# --- Training Function ---
def train_model():
    """Loads data, builds model, and runs the training loop."""
    logging.info("--- Starting PyTorch Training Process ---")

    # Load Data
    logging.info(f"Loading dataset: {config.DATASET_NAME}")
    ds = load_dataset_splits(dataset_name=config.DATASET_NAME)

    # Preprocess Data using Hugging Face `map`
    if feature_extractor is None:
        logging.error("Feature extractor not loaded. Exiting.")
        return

    logging.info("Applying feature extractor preprocessing...")
    ds = ds.rename_column("audio", "input_values")
    processed_datasets = ds.with_transform(preprocess_features)

    # Build Model
    logging.info(f"Building AST model for 2 classes.")
    model = build_transformer_model(num_classes=2, model_checkpoint=config.MODEL_CHECKPOINT)
    model.to(DEVICE) # Move model to GPU/CPU

    # Training Loop
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_SAVE_DIR, config.CHECKPOINT_FILENAME)

    trainer_output_dir = os.path.join(config.MODEL_SAVE_DIR, "trainer_output")
    os.makedirs(trainer_output_dir, exist_ok=True)

    logging.info("Defining TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=trainer_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LR,
        logging_steps=10,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        # fp16=True
    )

    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["valid"],
        processing_class=feature_extractor,
        compute_metrics=compute_metrics
    )

    logging.info("--- Starting Training with Hugging Face Trainer ---")
    trainer.train()

    logging.info("--- Trainer Training Finished ---")

    if training_args.load_best_model_at_end:
        logging.info(f"Saving best model to {model_save_path}")
        trainer.save_model(model_save_path)
        logging.info(f"Best model saved. Access final metrics and checkpoints at {training_args.output_dir}")

    if trainer.state.best_metric is not None:
        logging.info(f"Best validation accuracy achieved: {trainer.state.best_metric:.2f}% (if 'accuracy' was the best_metric)")
    else:
        logging.info("Best metric not available from trainer state.")
    logging.info(f"Training outputs (checkpoints, logs) are in {training_args.output_dir}")

# --- Main Execution ---
if __name__ == "__main__":
    train_model() 