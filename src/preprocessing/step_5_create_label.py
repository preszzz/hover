import logging
import yaml
import fnmatch
from pathlib import Path
import os
import shutil

# Assuming config.py is accessible
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the standard label filename
LABEL_FILENAME = "label.txt"

def load_label_mapping(mapping_path: Path) -> dict | None:
    """Loads the label mapping rules from the YAML file."""
    if not mapping_path.is_file():
        logging.error(f"Label mapping file not found: {mapping_path}")
        return None
    try:
        with open(mapping_path, 'r') as f:
            mapping = yaml.safe_load(f)
        logging.info(f"Successfully loaded label mapping from {mapping_path}")
        # Basic validation
        if not isinstance(mapping, dict):
            logging.error(f"Label mapping file {mapping_path} did not parse as a dictionary.")
            return None
        return mapping
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {mapping_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading label mapping file {mapping_path}: {e}")
        return None

def get_label_from_path(relative_path: Path, mapping_rules: dict) -> str:
    """Determines the label based on the file's relative path using mapping rules.
    
    Args:
        relative_path: Path relative to the interim directory
        mapping_rules: Dictionary of labeling rules from label_mapping.yaml
        
    Returns:
        str: The determined label
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
    
    # Use as_posix() for consistent forward slash separators in matching logic
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
                # Check if the file's relative path starts with the rule path
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

def create_label_files(processed_segments_dir: str, label_mapping_path: str):
    """Creates labels and moves features to final processed directory based on directory structure.
    
    Args:
        processed_segments_dir: Directory containing the processed segments
        label_mapping_path: Path to the label mapping YAML file
    """
    intermediate_segments_path = Path(processed_segments_dir)
    final_processed_path = Path(config.PROCESSED_DATA_DIR)
    final_processed_path.mkdir(parents=True, exist_ok=True)

    mapping_file_path = Path(label_mapping_path)
    mapping_rules = load_label_mapping(mapping_file_path)
    if mapping_rules is None:
        logging.critical("Failed to load label mapping. Aborting label creation.")
        return

    logging.info(f"Starting label file creation for segments in: {intermediate_segments_path}")
    label_files_created = 0
    segments_processed = 0
    segments_skipped_no_features = 0
    label_errors = 0

    # Find all directories containing features
    for segment_dir in intermediate_segments_path.rglob('*'):
        if not segment_dir.is_dir():
            continue
            
        # Check if this directory contains features
        feature_file = segment_dir / config.MFCC_FILENAME
        if not feature_file.is_file():
            segments_skipped_no_features += 1
            continue

        segments_processed += 1
        try:
            # Get relative path from intermediate directory
            relative_path = segment_dir.relative_to(intermediate_segments_path)
            
            # Determine label based on path
            label = get_label_from_path(relative_path, mapping_rules)
            label_dirname = label.replace(' ', '_').replace('/', '-')

            # Construct final path preserving dataset and chunk structure
            dataset_name = relative_path.parts[0]
            chunk_name = segment_dir.name
            final_chunk_dir = final_processed_path / dataset_name / label_dirname / chunk_name

            # Create final directory and move features
            final_chunk_dir.mkdir(parents=True, exist_ok=True)

            # Move feature files
            moved_files = 0
            for feature_filename in [config.MFCC_FILENAME, config.SIGNAL_FILENAME]:
                source_feature_path = segment_dir / feature_filename
                dest_feature_path = final_chunk_dir / feature_filename
                if source_feature_path.is_file():
                    shutil.move(str(source_feature_path), str(dest_feature_path))
                    logging.debug(f"Moved {source_feature_path.name} to {dest_feature_path}")
                    moved_files += 1
                else:
                    logging.warning(f"Feature file {feature_filename} not found in {segment_dir}")

            if moved_files == 0:
                logging.error(f"No feature files found to move from {segment_dir}")
                label_errors += 1
                continue

            # Write label file
            label_file_path = final_chunk_dir / LABEL_FILENAME
            with open(label_file_path, 'w') as f:
                f.write(label)
            label_files_created += 1
            logging.debug(f"Wrote label '{label}' to {label_file_path}")

            # Clean up intermediate directory if empty
            try:
                if not any(segment_dir.iterdir()):
                    os.rmdir(segment_dir)
                    logging.debug(f"Removed empty intermediate directory: {segment_dir}")
                else:
                    logging.warning(f"Intermediate directory not empty, not removing: {segment_dir}")
            except OSError as e:
                logging.warning(f"Could not remove intermediate directory {segment_dir}: {e}")

        except Exception as e:
            logging.error(f"Error processing segment directory {segment_dir}: {e}")
            label_errors += 1

    logging.info(f"Label file creation & final move complete:")
    logging.info(f"Segments Processed: {segments_processed}")
    logging.info(f"Label Files Created & Features Moved: {label_files_created}")
    logging.info(f"Skipped (No Features): {segments_skipped_no_features}")
    logging.info(f"Errors: {label_errors}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    mapping_file = script_dir / 'label_mapping.yaml'
    create_label_files(config.INTERIM_SPLIT_DIR, str(mapping_file))
