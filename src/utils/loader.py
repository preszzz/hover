"""Path management utilities for the preprocessing pipeline."""

import logging
from pathlib import Path
import shutil
import yaml
import fnmatch
from datasets import load_dataset, Dataset, Audio
import config

def load_dataset_splits(dataset_name: str) -> Dataset:
    """Load a dataset from the Hugging Face Hub.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Hugging Face Dataset object
    """
    try:
        print(f"Loading dataset {dataset_name} to {config.CACHE_DIR}")
        dataset = load_dataset(dataset_name, cache_dir=config.CACHE_DIR)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=config.TARGET_SAMPLE_RATE, mono=True))
        
        logging.info(f"Dataset loaded successfully with splits: {list(dataset.keys())}")
        logging.info(f"Train split size: {dataset['train'].num_rows}")
        if 'valid' in dataset:
            logging.info(f"Validation split size: {dataset['valid'].num_rows}")
        if 'test' in dataset:
            logging.info(f"Test split size: {dataset['test'].num_rows}")

        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset {dataset_name}: {e}")
        raise

def clean_directory(path: Path) -> bool:
    """Remove a directory and all its contents recursively.
    
    Args:
        path: Directory path to remove
        
    Returns:
        True if removal was successful, False on error
    """
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception as e:
        logging.error(f"Failed to clean directory {path}: {e}")

def load_label_mapping(mapping_path: Path) -> dict | None:
    """Load label mapping rules from YAML file.
    
    Args:
        mapping_path: Path to the label mapping YAML file
        
    Returns:
        Dictionary of mapping rules or None if loading fails
    """
    if not mapping_path.is_file():
        logging.error(f"Label mapping file not found: {mapping_path}")
        return None
        
    try:
        with open(mapping_path, 'r') as f:
            mapping = yaml.safe_load(f)
            
        # Basic validation
        if not isinstance(mapping, dict):
            logging.error(f"Label mapping file {mapping_path} did not parse as a dictionary.")
            return None
            
        logging.info(f"Successfully loaded label mapping from {mapping_path}")
        return mapping
        
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {mapping_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading label mapping file {mapping_path}: {e}")
        return None

def get_label(relative_path: Path, mapping_rules: dict) -> str:
    """Determine label based on file's relative path using mapping rules.
    First part of path is treated as dataset name.
    
    Args:
        relative_path: Path relative to the interim directory (includes dataset name)
        mapping_rules: Dictionary of labeling rules from label_mapping.yaml
        
    Returns:
        Determined label string, 'unknown' if no rules match
    """
    # First part of the path is the dataset name
    dataset_name = relative_path.parts[0]
    
    # Get the path relative to the dataset directory
    relative_path_in_dataset = Path(*relative_path.parts[1:])
    
    dataset_rules = mapping_rules.get(dataset_name)
    if not dataset_rules:
        logging.warning(f"No rules found for dataset '{dataset_name}' in mapping file. Using 'unknown'.")
        return "unknown"

    rules = dataset_rules.get('rules', [])
    default_label = dataset_rules.get('default_label', 'unknown')
    
    # Use as_posix() for consistent forward slash separators
    relative_path_str = relative_path_in_dataset.as_posix()

    for rule in rules:
        rule_type = rule.get('type')
        label = rule.get('label')
        if not (rule_type and label):
            logging.warning(f"Skipping invalid rule in {dataset_name}: {rule}")
            continue

        try:
            if rule_type == 'subdir':
                rule_path_str = rule.get('path')
                if not rule_path_str:
                    logging.warning(f"Skipping subdir rule with no path in {dataset_name}: {rule}")
                    continue
                # Ensure rule path uses forward slashes and ends with one for prefix matching
                rule_path_prefix = (Path(rule_path_str).as_posix() + '/') if not rule_path_str.endswith('/') else Path(rule_path_str).as_posix()
                if relative_path_str.startswith(rule_path_prefix):
                    logging.debug(f"Matched subdir rule '{rule_path_prefix}' for '{relative_path_str}' -> {label}")
                    return label
                    
            elif rule_type == 'pattern':
                rule_glob = rule.get('glob')
                if not rule_glob:
                    logging.warning(f"Skipping pattern rule with no glob in {dataset_name}: {rule}")
                    continue
                if fnmatch.fnmatch(relative_path_in_dataset.name, rule_glob):
                    logging.debug(f"Matched pattern rule '{rule_glob}' for '{relative_path_in_dataset.name}' -> {label}")
                    return label
                    
            else:
                logging.warning(f"Unknown rule type '{rule_type}' in {dataset_name}: {rule}")
                
        except Exception as e:
            logging.error(f"Error applying rule {rule} to {relative_path_str} in {dataset_name}: {e}")

    logging.debug(f"No specific rule matched for '{relative_path_str}' in {dataset_name}. Using default: {default_label}")
    return default_label


