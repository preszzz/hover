#!/usr/bin/env python
"""Hyperparameter tuning script for the audio classification model using Optuna."""

import logging
import sys
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from project modules
import config
from src.utils.loader import load_dataset_splits
from src.feature_engineering.feature_loader import preprocess_features, feature_extractor
from src.models.transformer_model import build_transformer_model
from src.training.train import get_device, collate_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get device for training
DEVICE = get_device()

def objective(trial):
    """Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Validation accuracy to maximize (implicitly float)
    """
    # Define hyperparameters to optimize
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
    
    # Log the parameters for this trial
    logging.info(f"Trial #{trial.number} - Parameters: lr={lr}, batch_size={batch_size}, "
                f"freeze_backbone={freeze_backbone}, dropout_rate={dropout_rate}, "
                f"weight_decay={weight_decay}, optimizer={optimizer_name}")
    
    # Load data
    datasets = load_dataset_splits(dataset_name=config.DATASET_NAME)
    
    # Check if feature extractor is loaded
    if feature_extractor is None:
        logging.error("Feature extractor not loaded. Exiting.")
        return 0.0
    
    # Preprocess data using Hugging Face `map`
    processed_datasets = datasets.map(
        preprocess_features,
        batched=True,
        remove_columns=['audio']
    )
    
    # Create DataLoaders with trial batch size
    train_dataloader = DataLoader(
        processed_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        processed_datasets["valid"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Build model
    model = build_transformer_model(num_classes=2, model_checkpoint=config.MODEL_CHECKPOINT)
    
    # Apply dropout rate to attention dropout
    if hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = dropout_rate
    
    # Freeze backbone if specified
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
                
    model.to(DEVICE)
    
    # Setup optimizer based on trial suggestion
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Number of epochs for hyperparameter tuning (reduced from full training)
    num_epochs = 5
    
    # Storage for best validation accuracy
    best_val_accuracy = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, batch in enumerate(train_dataloader):
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
            
            # Report intermediate results
            if (i + 1) % 50 == 0:
                logging.info(f"Trial #{trial.number} - Epoch {epoch+1}/{num_epochs}, "
                            f"Batch {i+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
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
        
        # Log epoch results
        logging.info(f"Trial #{trial.number} - Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Report intermediate value to Optuna
        trial.report(val_accuracy, epoch)
        
        # Handle pruning based on intermediate results
        if trial.should_prune():
            logging.info(f"Trial #{trial.number} pruned.")
            raise optuna.exceptions.TrialPruned()
        
        # Update best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
    
    # Return best validation accuracy for this trial
    return best_val_accuracy


def run_hyperparameter_tuning(n_trials: int, study_name: str):
    """Run the hyperparameter tuning process.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the study
    """
    logging.info(f"Starting hyperparameter tuning with {n_trials} trials")
    
    logging.info("Creating new study")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=study_name
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print best trial information
    logging.info("Best trial:")
    best_trial = study.best_trial
    logging.info(f"Value (Validation Accuracy): {best_trial.value:.2f}%")
    logging.info("Params:")
    for key, value in best_trial.params.items():
        logging.info(f"{key}: {value}")


if __name__ == "__main__":
    # Default values for trials and study name
    N_TRIALS = 20
    STUDY_NAME = "ast_hyperparameter_tuning"

    run_hyperparameter_tuning(n_trials=N_TRIALS, study_name=STUDY_NAME) 