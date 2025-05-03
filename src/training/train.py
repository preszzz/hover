"""Main script for training the audio classification model."""

import logging
import os
import tensorflow as tf
from datetime import datetime

# Local imports (adjust paths as necessary)
from src.feature_engineering.feature_loader import get_feature_loaders, preprocess_features
from src.models.cnn_model import build_cnn_model
# from src.models.lstm_model import build_lstm_model # Uncomment when ready
# from src.models.transformer_model import build_transformer_model # Uncomment when ready

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (Consider moving to a central config file or using argparse) ---
MODEL_TYPE = 'cnn' # Options: 'cnn', 'lstm', 'transformer'
NUM_CLASSES = 2 # Binary classification (drone vs no-drone)
HUGGINGFACE_DATASET_ID = "hover-d/test"

# Training Hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Input Shape (IMPORTANT: Must match the output of preprocess_features)
# Example for CNN with stacked MFCC, delta, delta2:
INPUT_SHAPE = (44, 40, 3) # (time_frames, mfcc_coeffs, channels)
# Example for LSTM/Transformer (if features are flattened per time step):
# INPUT_SHAPE = (44, 120) # (time_steps, num_features)

# Model Saving
MODEL_SAVE_DIR = "./trained_models"

def train_model():
    """Loads data, builds model, trains, and saves it."""
    
    logging.info("--- Starting Model Training --- ")
    logging.info(f"Model Type: {MODEL_TYPE}")
    logging.info(f"Dataset: {HUGGINGFACE_DATASET_ID}")
    logging.info(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")

    # 1. Load Data
    logging.info("Loading datasets...")
    train_ds, valid_ds, _ = get_feature_loaders(dataset_id=HUGGINGFACE_DATASET_ID)
    
    # 2. Preprocess Data (Apply the preprocessing function)
    #    This step needs to be fully implemented in feature_loader.py
    logging.info("Applying preprocessing...")
    # IMPORTANT: Ensure preprocess_features reshapes/selects data matching INPUT_SHAPE
    # train_ds_processed = train_ds.map(preprocess_features, batched=True, num_parallel_calls=tf.data.AUTOTUNE)
    # valid_ds_processed = valid_ds.map(preprocess_features, batched=True, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Placeholder: Use raw datasets for now - REMOVE when preprocess_features is ready
    logging.warning("Using RAW datasets - Preprocessing function needs implementation!")
    train_ds_processed = train_ds 
    valid_ds_processed = valid_ds
    # End Placeholder

    # Convert to TensorFlow datasets (example - adapt as needed)
    # This conversion depends heavily on how preprocess_features returns data
    # You might need to adjust feature keys ('features', 'label')
    def format_for_tf(batch):
        # Placeholder: assumes preprocess_features returns suitable format
        # Modify this based on actual output of preprocess_features
        # features = tf.cast(batch['features'], tf.float32)
        # labels = tf.cast(batch['label'], tf.int32)
        # return features, labels
        pass # Replace with actual TF formatting

    # tf_train_ds = train_ds_processed.map(format_for_tf).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # tf_valid_ds = valid_ds_processed.map(format_for_tf).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    logging.warning("TensorFlow dataset conversion needs implementation based on preprocessing output.")
    tf_train_ds = None # Replace with actual TF dataset
    tf_valid_ds = None # Replace with actual TF dataset

    # 3. Build Model
    logging.info(f"Building {MODEL_TYPE} model...")
    model = None
    if MODEL_TYPE == 'cnn':
        model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    elif MODEL_TYPE == 'lstm':
        # model = build_lstm_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
        logging.error("LSTM model training not implemented yet.")
        return
    elif MODEL_TYPE == 'transformer':
        # model = build_transformer_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
        logging.error("Transformer model training not implemented yet.")
        return
    else:
        logging.error(f"Unknown model type: {MODEL_TYPE}")
        return

    if model is None:
        logging.error("Model building failed.")
        return
        
    # Adjust learning rate if needed (overrides default in model build)
    # tf.keras.backend.set_value(model.optimizer.learning_rate, LEARNING_RATE)

    # 4. Train Model
    if tf_train_ds is None or tf_valid_ds is None:
        logging.error("Training cannot proceed without valid TensorFlow datasets. Implement preprocessing and TF conversion.")
        return
        
    logging.info("Starting training...")
    # Add callbacks (e.g., ModelCheckpoint, EarlyStopping, TensorBoard)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_filename = f"{MODEL_TYPE}_model_{timestamp}.keras" # Use .keras format
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True, # Save only the best model based on validation performance
            monitor='val_loss', # Metric to monitor (e.g., val_accuracy)
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, # Number of epochs with no improvement after which training will be stopped
            verbose=1,
            restore_best_weights=True # Restore model weights from the epoch with the best value
        ),
        # tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{MODEL_TYPE}_{timestamp}") # Optional: For TensorBoard visualization
    ]
    
    history = model.fit(
        tf_train_ds,
        epochs=EPOCHS,
        validation_data=tf_valid_ds,
        callbacks=callbacks
    )

    logging.info("Training finished.")
    logging.info(f"Best model saved to: {model_save_path}")

    # Optional: Plot training history (Accuracy, Loss)
    # import matplotlib.pyplot as plt
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # ... add plots for loss ...
    # plt.show()

    logging.info("--- Model Training Complete --- ")

if __name__ == "__main__":
    train_model() 