#!/usr/bin/env python
"""Evaluate the trained audio classification model using PyTorch."""

import logging
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import from project modules
import config
from utils import load_dataset_splits
from feature_engineering.feature_loader import preprocess_features, feature_extractor
from models.transformer_model import build_transformer_model
# Import collate_fn and device from train script (or redefine if preferred)
from train import collate_fn, get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---

MODEL_LOAD_DIR = os.path.join(config.ROOT_DIR, 'output_models')
MODEL_FILENAME = "ast_best_model" # The saved model state dict

RESULTS_SAVE_DIR = os.path.join(config.ROOT_DIR, 'evaluation_results_pytorch')
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
CLASSIFICATION_REPORT_FILENAME = "classification_report.txt"

# --- Device Setup ---
DEVICE = get_device() # Use the same device logic as training

# --- Evaluation Function ---
def evaluate_model():
    """Loads test data, loads trained model, runs inference, and calculates metrics."""
    logging.info("--- Starting PyTorch Model Evaluation ---")

    # 1. Load Test Data
    logging.info(f"Loading dataset: {config.DATASET_NAME}")
    datasets = load_dataset_splits(dataset_name=config.DATASET_NAME)
    if "test" not in datasets:
        logging.error("Test split not found in the dataset. Exiting.")
        return
    test_dataset = datasets["test"]

    # 2. Preprocess Test Data
    if feature_extractor is None:
        logging.error("Feature extractor not loaded. Cannot preprocess. Exiting.")
        return
    logging.info("Applying feature extractor preprocessing to test set...")
    processed_test_dataset = test_dataset.map(
        preprocess_features,
        batched=True,
        remove_columns=['audio']
    )

    # 3. Create Test DataLoader
    logging.info("Creating Test DataLoader...")
    test_dataloader = DataLoader(
        processed_test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # Important: Do not shuffle test data
        collate_fn=collate_fn
    )

    # 4. Load Trained Model
    model_path = os.path.join(MODEL_LOAD_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        logging.error(f"Model state dict not found at {model_path}. Train the model first.")
        return

    logging.info(f"Loading trained model state dict from: {model_path}")
    model = build_transformer_model(num_classes=2, model_checkpoint=config.MODEL_CHECKPOINT)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logging.info("Model state dict loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model state dict: {e}", exc_info=True)
        return

    model.to(DEVICE)
    model.eval() # Set model to evaluation mode

    # 5. Run Inference
    logging.info("Running inference on the test set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_values = batch["input_values"] # Already on DEVICE via collate_fn
            labels = batch["labels"]         # Already on DEVICE via collate_fn

            outputs = model(input_values=input_values)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy()) # Move predictions to CPU for numpy/sklearn
            all_labels.extend(labels.cpu().numpy())    # Move labels to CPU

    logging.info("Inference complete.")

    # Ensure we have predictions and labels
    if not all_labels or not all_preds:
        logging.error("No labels or predictions collected. Cannot evaluate.")
        return

    # 6. Calculate and Report Metrics
    logging.info("--- Evaluation Results ---")
    accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["No", "Yes"], # Use class names from feature_loader
        digits=4
    )
    logging.info("\nClassification Report:\n")
    print(report) # Print to console for immediate visibility

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    logging.info("\nConfusion Matrix:\n")
    print(cm)

    # 7. Save Results
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    # Save classification report to file
    report_path = os.path.join(RESULTS_SAVE_DIR, CLASSIFICATION_REPORT_FILENAME)
    try:
        with open(report_path, "w") as f:
            f.write(f"Evaluation Results for model: {model_path}\n")
            f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(np.array2string(cm))
        logging.info(f"Classification report saved to {report_path}")
    except Exception as e:
        logging.error(f"Failed to save classification report: {e}")

    # Plot and save confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(RESULTS_SAVE_DIR, CONFUSION_MATRIX_FILENAME)
        plt.savefig(cm_path)
        plt.close() # Close the plot to free memory
        logging.info(f"Confusion matrix plot saved to {cm_path}")
    except Exception as e:
        logging.error(f"Failed to plot or save confusion matrix: {e}", exc_info=True)

    logging.info("--- Evaluation Finished ---")

# --- Main Execution ---
if __name__ == "__main__":
    evaluate_model() 