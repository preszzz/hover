"""Path management utilities for the preprocessing pipeline."""

import logging
from pathlib import Path
import shutil
def get_final_chunk_path(base_dir: Path, dataset: str, label: str, chunk_name: str) -> Path:
    """Generate path for final chunk directory.
    
    Args:
        base_dir: Base directory for processed data
        dataset: Dataset name
        label: Label string (will be sanitized)
        chunk_name: Name of the chunk
        
    Returns:
        Path object for the final chunk directory
    """
    # Sanitize label for filesystem
    label_dirname = label.replace(' ', '_').replace('/', '-')
    return base_dir / dataset / label_dirname / chunk_name

def clean_directory(path: Path) -> bool:
    """Remove an empty directory.
    
    Args:
        path: Directory path to remove
        
    Returns:
        True if removal was successful, False on error
    """
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception as e:
        logging.error(f"Failed to remove directory {path}: {e}")
