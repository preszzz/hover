#!/usr/bin/env python
"""Main training script for the audio classification model using PyTorch."""

import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from project modules
import config
from src.utils import load_dataset_splits
from src.feature_engineering.feature_loader import preprocess_features, feature_extractor
from src.models.transformer_model import build_transformer_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Device Setup ---
def get_device():
    if torch.cuda.is_available():
        logging.info("CUDA available. Using GPU.")
        return torch.device("cuda")
    # Add check for MPS (MacOS Metal Performance Shaders) if relevant
    elif torch.backends.mps.is_available():
        logging.info("MPS available. Using GPU.")
        return torch.device("mps")
    else:
        logging.info("CUDA/MPS not available. Using CPU.")
        return torch.device("cpu")

DEVICE = get_device()

# --- Data Collator ---
def collate_fn(batch):
    """Prepares batches for the PyTorch DataLoader.
    Converts NumPy arrays from the dataset mapping step to PyTorch tensors.
    Moves tensors to the specified device.
    """
    # Extract features and labels from the batch
    # The feature extractor output key is expected to be 'input_values'
    # after the preprocess_features mapping.
    input_values = [item['input_values'] for item in batch]
    labels = [item['label'] for item in batch]

    # Convert to PyTorch tensors
    # AST expects input_values with shape (batch_size, num_mel_bins, time_frames)
    inputs = torch.tensor(input_values, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "input_values": inputs, # AST models expect 'input_values' key
        "labels": labels
    }

# --- Training Function ---
def train_model():
    """Loads data, builds model, and runs the training loop."""
    logging.info("--- Starting PyTorch Training Process ---")

    # 1. Load Data
    logging.info(f"Loading dataset: {config.DATASET_NAME}")
    datasets = load_dataset_splits(dataset_name=config.DATASET_NAME)

    # 2. Preprocess Data using Hugging Face `map`
    if feature_extractor is None:
        logging.error("Feature extractor not loaded. Exiting.")
        return

    logging.info("Applying feature extractor preprocessing...")
    processed_datasets = datasets.map(
        preprocess_features,
        batched=True,
        remove_columns=['audio'] # Keep only label and processed features
        # Consider adding num_proc=os.cpu_count() for parallel processing if needed
    )
    # After mapping, the features should be under the key 'input_values'
    # The collate_fn expects this key.

    # 3. Create DataLoaders
    logging.info("Creating DataLoaders...")
    train_dataloader = DataLoader(
        processed_datasets["train"],
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        processed_datasets["valid"],
        batch_size=config.BATCH_SIZE,
        shuffle=False, # No need to shuffle validation data
        collate_fn=collate_fn
    )

    # 4. Build Model
    logging.info(f"Building AST model for 2 classes.")
    model = build_transformer_model(num_classes=2, model_checkpoint=config.MODEL_CHECKPOINT)
    model.to(DEVICE) # Move model to GPU/CPU

    # 5. Define Optimizer and Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss() # Standard loss for multi-class classification

    # 6. Training Loop
    best_val_accuracy = 0.0
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_SAVE_DIR, config.CHECKPOINT_FILENAME)

    logging.info("--- Starting Training Loop ---")
    for epoch in range(config.EPOCHS):
        logging.info(f"epoch {epoch+1}/{config.EPOCHS}")

        # --- Training Phase ---
        model.train() # Set model to training mode
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, batch in enumerate(train_dataloader):
            # Data is already on DEVICE via collate_fn
            input_values = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_values=input_values)
            logits = outputs.logits

            # Calculate loss
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            total_train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (i + 1) % 50 == 0: # Log progress every 50 batches
                logging.info(f"Batch {i+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = 100 * train_correct / train_total
        logging.info(f"End of Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad(): # Disable gradient calculations
            for batch in val_dataloader:
                input_values = batch["input_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                # Forward pass
                outputs = model(input_values=input_values)
                logits = outputs.logits

                # Calculate loss
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = 100 * val_correct / val_total
        logging.info(f"  End of Epoch {epoch+1} - Avg Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # --- Checkpoint Saving ---
        if val_accuracy > best_val_accuracy:
            logging.info(f"Validation accuracy improved ({best_val_accuracy:.2f}% -> {val_accuracy:.2f}%). Saving model...")
            best_val_accuracy = val_accuracy
            # Save only the model state_dict
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved to {model_save_path}")
        else:
            logging.info(f"Validation accuracy did not improve from {best_val_accuracy:.2f}%")

    logging.info("--- Training Finished ---")
    logging.info(f"Best validation accuracy achieved: {best_val_accuracy:.2f}%" )
    logging.info(f"Best model state dict saved to {model_save_path}")

# --- Main Execution ---
if __name__ == "__main__":
    train_model() 