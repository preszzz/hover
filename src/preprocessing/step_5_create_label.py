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
# Define the metadata filename (must match step_3)
SOURCE_INFO_FILENAME = "source_info.yaml"


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


def get_label_for_source_path(dataset_name: str, relative_path_in_dataset: Path, mapping_rules: dict) -> str:
    """Determines the label for a file based on its dataset and relative path, using mapping rules."""
    dataset_rules = mapping_rules.get(dataset_name)
    if not dataset_rules:
        logging.warning(f"No rules found for dataset '{dataset_name}' in mapping file. Using 'unknown'.")
        return "unknown"

    rules = dataset_rules.get('rules', [])
    default_label = dataset_rules.get('default_label', 'unknown')
    filename = relative_path_in_dataset.name
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
                if fnmatch.fnmatch(filename, rule_glob):
                    logging.debug(f"Matched pattern rule '{rule_glob}' for '{filename}' -> {label}")
                    return label
            else:
                 logging.warning(f"Unknown rule type '{rule_type}' in {dataset_name}: {rule}")
        except Exception as e:
            logging.error(f"Error applying rule {rule} to {relative_path_str} in {dataset_name}: {e}")

    logging.debug(f"No specific rule matched for '{relative_path_str}' in {dataset_name}. Using default: {default_label}")
    return default_label


def create_label_files(processed_segments_dir: str, wav_conversion_dir: str, label_mapping_path: str):
    """Scans intermediate segments, determines labels, moves features+creates label in final processed dir."""
    # Rename variable for clarity - this is the intermediate source
    intermediate_segments_path = Path(processed_segments_dir)
    final_processed_path = Path(config.PROCESSED_DATA_DIR)
    # Ensure the final base processed directory exists
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

    segment_dirs = []
    # Iterate through ALL directories in intermediate_segments_path that look like chunk dirs
    for item in intermediate_segments_path.rglob('*_chunk_*'):
        if item.is_dir():
            segment_dirs.append(item)

    logging.info(f"Found {len(segment_dirs)} potential segment directories under {intermediate_segments_path}...")

    for segment_dir in segment_dirs:
        segments_processed += 1

        # Check if feature extraction was successful for this segment
        feature_file = segment_dir / config.MFCC_FILENAME
        if not feature_file.is_file():
            segments_skipped_no_features += 1
            continue

        try:
            # Read Source Info Metadata
            metadata_path = segment_dir / SOURCE_INFO_FILENAME
            if not metadata_path.is_file():
                logging.error(f"Metadata file {SOURCE_INFO_FILENAME} not found in {segment_dir}. Cannot determine label.")
                label_errors += 1
                continue

            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)

            if not metadata or 'dataset_name' not in metadata or 'relative_source_path_in_dataset' not in metadata:
                logging.error(f"Invalid or incomplete metadata in {metadata_path}. Cannot determine label.")
                label_errors += 1
                continue

            dataset_name = metadata['dataset_name']
            # Convert back to Path object for get_label_for_source_path
            relative_source_path_in_dataset = Path(metadata['relative_source_path_in_dataset'])

            # Determine Label
            label = get_label_for_source_path(dataset_name, relative_source_path_in_dataset, mapping_rules)
            # Sanitize label for use in directory name
            label_dirname = label.replace(' ', '_').replace('/', '-')

            # Construct Final Path
            chunk_name = segment_dir.name  # e.g., "c_chunk_1"
            final_chunk_dir = final_processed_path / dataset_name / label_dirname / chunk_name

            # Create final dir, Move Features, Write Label
            try:
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
                        logging.warning(f"Feature file {feature_filename} not found in intermediate dir {segment_dir}")

                if moved_files == 0:
                     logging.error(f"No feature files found to move from {segment_dir}. Skipping label/cleanup.")
                     label_errors += 1
                     continue  # Don't write label or cleanup if no features were moved

                # Write Label File to FINAL location
                label_file_path = final_chunk_dir / LABEL_FILENAME
                with open(label_file_path, 'w') as f:
                    f.write(label)
                label_files_created += 1
                logging.debug(f"Wrote label '{label}' to {label_file_path}")

                # Clean up metadata file and intermediate dir AFTER successful move/label
                try:
                    os.remove(metadata_path)
                    logging.debug(f"Removed intermediate metadata file: {metadata_path}")
                except OSError as e:
                    logging.warning(f"Could not remove intermediate metadata file {metadata_path}: {e}")

                # Try removing the intermediate segment directory (should be empty now)
                try:
                    # Check if empty first (safer)
                    if not any(segment_dir.iterdir()):
                         os.rmdir(segment_dir)
                         logging.debug(f"Removed empty intermediate segment directory: {segment_dir}")
                    else:
                        logging.warning(f"Intermediate segment directory not empty, not removing: {segment_dir}")
                except OSError as e:
                    logging.warning(f"Could not remove intermediate segment directory {segment_dir}: {e}")

            except Exception as e:
                logging.error(f"Error during final move/label/cleanup for {segment_dir} -> {final_chunk_dir}: {e}", exc_info=True)
                label_errors += 1

        except Exception as e:
            logging.error(f"Error processing intermediate segment directory {segment_dir}: {e}", exc_info=True)
            label_errors += 1

    logging.info(f"Label file creation & final move finished. Total Intermediate Segment Dirs Found: {len(segment_dirs)}")
    logging.info(f"Label Files Created & Features Moved: {label_files_created}")
    logging.info(f"Segments Processed: {segments_processed}, Skipped (No Features in Intermediate): {segments_skipped_no_features}")
    logging.info(f"Errors During Labeling/Move/Cleanup: {label_errors}")


if __name__ == "__main__":
    # Determine the path to the label mapping file relative to this script
    script_dir = Path(__file__).parent
    mapping_file = script_dir / 'label_mapping.yaml'

    # Ensure required INTERIM config path exists
    interim_dir_path = Path(config.INTERIM_SPLIT_DIR)

    if not interim_dir_path.is_dir():
        logging.error(f"Intermediate split directory not found: {interim_dir_path}")
        logging.error("Ensure Steps 1-3 of the pipeline have run successfully.")
    else:
        create_label_files(config.INTERIM_SPLIT_DIR,
                           config.WAV_CONVERSION_DIR,
                           str(mapping_file))
 