#!/usr/bin/env python
"""Evaluate the trained audio classification model"""

import os
import logging
from transformers import TrainingArguments, Trainer, ASTForAudioClassification, ASTFeatureExtractor

# Import from project modules
import config
from utils import load_dataset_splits
from utils.hardware import get_device
from utils.metric import compute_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
RESULTS_DIR = os.path.join(config.ROOT_DIR, 'evaluation_results')
EVALUATION_RESULTS = "evaluation_summary.txt"

# --- Device Setup ---
DEVICE = get_device()

# --- Evaluation Function ---
def evaluate_model():
    """Loads test data, loads trained model, runs evaluation using Trainer."""
    logging.info("--- Starting Model Evaluation ---")

    # 1. Load Test Data
    datasets = load_dataset_splits(dataset_name=config.DATASET_NAME)
    if "test" not in datasets:
        logging.error("Test split not found in the dataset. Exiting.")
        return
    test_dataset = datasets["test"]
    test_dataset = test_dataset.rename_column("audio", "input_values")

    # 2. Load Trained Model
    try:
        model = ASTForAudioClassification.from_pretrained(config.MODEL_HUB_ID, cache_dir=config.CACHE_DIR)
        feature_extractor = ASTFeatureExtractor.from_pretrained(config.MODEL_HUB_ID, sampling_rate=config.TARGET_SAMPLE_RATE)
        logging.info("Model and feature extractor loaded successfully.")
        model.to(DEVICE)
    except Exception as e:
        logging.error(f"Failed to load model and feature extractor: {e}", exc_info=True)
        return

    # 3. Preprocess Test Data
    def preprocess_features(example):
        audio_arrays = [x["array"] for x in example['input_values']]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=config.TARGET_SAMPLE_RATE,
            max_length=config.CHUNK_LENGTH_SAMPLES,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        example['input_values'] = inputs.input_values
        return example

    logging.info("Applying feature extractor preprocessing to test set...")
    processed_test_dataset = test_dataset.with_transform(preprocess_features)

    # 4. Setup TrainingArguments & Trainer for Evaluation
    logging.info("Setting up Trainer for evaluation...")
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=processed_test_dataset,
        compute_metrics=compute_metrics,
        processing_class=feature_extractor
    )

    # 5. Run Evaluation
    logging.info("Starting evaluation...")
    eval_results = trainer.evaluate()
    logging.info("Evaluation complete.")

    # 6. Report and Save Metrics
    logging.info("--- Evaluation Results ---")
    for key, value in eval_results.items():
        logging.info(f"{key}: {value}")
        print(f"{key}: {value}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, EVALUATION_RESULTS)
    try:
        with open(results_path, "w") as f:
            f.write(f"Evaluation Results for model: {config.MODEL_HUB_ID}\n\n")
            for key, value in eval_results.items():
                f.write(f"{key}: {value}\n")
        logging.info(f"Evaluation summary saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save evaluation summary: {e}")

    logging.info("--- Evaluation Finished ---")

# --- Main Execution ---
if __name__ == "__main__":
    evaluate_model() 