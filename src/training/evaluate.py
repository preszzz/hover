"""Evaluates a trained model on the test dataset."""

import logging
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
from src.feature_engineering.feature_loader import load_data, preprocess_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (Consider moving to a central config file or using argparse) ---
HUGGINGFACE_DATASET_ID = "hover-d/test"
MODEL_PATH = "./trained_models/cnn_model_YYYYMMDD-HHMMSS.keras" # IMPORTANT: Replace with the actual path to your trained model
BATCH_SIZE = 64
NUM_CLASSES = 2 # Should match training configuration
CLASS_NAMES = ['No Drone', 'Drone'] # Adjust if you have different class names/order

def evaluate_model(model_path: str):
    """Loads a trained model and evaluates it on the test set."""
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return

    logging.info(f"--- Starting Model Evaluation --- ")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Dataset: {HUGGINGFACE_DATASET_ID}")

    # 1. Load Test Data
    logging.info("Loading test dataset...")
    test_ds = load_data('test', dataset_id=HUGGINGFACE_DATASET_ID)

    # 2. Preprocess Test Data
    logging.info("Applying preprocessing to test data...")
    # IMPORTANT: Use the *exact same* preprocessing as during training
    # test_ds_processed = test_ds.map(preprocess_features, batched=True, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Placeholder: Use raw datasets for now - REMOVE when preprocess_features is ready
    logging.warning("Using RAW test dataset - Preprocessing function needs implementation!")
    test_ds_processed = test_ds 
    # End Placeholder

    # Convert to TensorFlow dataset (using the same logic as in train.py)
    def format_for_tf(batch):
        # Placeholder - Needs implementation based on preprocess_features output
        # features = tf.cast(batch['features'], tf.float32)
        # labels = tf.cast(batch['label'], tf.int32)
        # return features, labels
        pass
    
    # tf_test_ds = test_ds_processed.map(format_for_tf).batch(BATCH_SIZE)
    logging.warning("TensorFlow test dataset conversion needs implementation.")
    tf_test_ds = None # Replace with actual TF dataset

    if tf_test_ds is None:
        logging.error("Evaluation cannot proceed without a valid TensorFlow test dataset.")
        return

    # 3. Load Trained Model
    logging.info(f"Loading trained model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully.")
        model.summary()
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        return

    # 4. Evaluate Model
    logging.info("Evaluating model on the test set...")
    # results = model.evaluate(tf_test_ds, verbose=1)
    # loss = results[0]
    # accuracy = results[1]
    # logging.info(f"Test Loss: {loss:.4f}")
    # logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.warning("Actual evaluation using model.evaluate() is commented out.")
    loss, accuracy = 0.0, 0.0 # Placeholder values

    # 5. Generate Predictions and Detailed Report
    logging.info("Generating predictions for detailed report...")
    # y_pred_probs = model.predict(tf_test_ds)
    
    # # Extract true labels (requires iterating through the tf.data.Dataset)
    # y_true = []
    # for _, labels in tf_test_ds.unbatch(): # Unbatch to get individual labels
    #     y_true.append(labels.numpy())
    # y_true = np.array(y_true)
    
    # Convert probabilities to class predictions
    if NUM_CLASSES == 2:
        # y_pred = (y_pred_probs > 0.5).astype(int).flatten() # For binary sigmoid output
        pass # Placeholder
    else:
        # y_pred = np.argmax(y_pred_probs, axis=1) # For multi-class softmax output
        pass # Placeholder

    logging.warning("Prediction generation and true label extraction need implementation.")
    y_true = None # Replace with actual labels
    y_pred = None # Replace with actual predictions

    if y_true is not None and y_pred is not None:
        logging.info("\n--- Classification Report --- ")
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
        print(report)

        logging.info("\n--- Confusion Matrix --- ")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        # You might want to save the plot instead of showing it directly
        # plt.savefig(f"./confusion_matrix_{os.path.basename(model_path)}.png")
        plt.show() 
    else:
        logging.warning("Cannot generate classification report or confusion matrix without predictions and true labels.")

    logging.info("--- Model Evaluation Complete --- ")

if __name__ == "__main__":
    # IMPORTANT: Update MODEL_PATH before running!
    if "YYYYMMDD-HHMMSS" in MODEL_PATH:
        logging.error("Please update the MODEL_PATH variable in the script before running evaluation.")
    else:
        evaluate_model(model_path=MODEL_PATH) 