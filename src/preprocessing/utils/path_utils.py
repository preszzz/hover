"""Path management utilities for the preprocessing pipeline."""

import logging
from pathlib import Path

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

def ensure_directory(path: Path) -> bool:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        True if directory exists or was created, False on error
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        return False

def clean_directory(path: Path) -> bool:
    """Remove directory if it exists and is empty.
    
    Args:
        path: Directory path to clean
        
    Returns:
        True if directory was removed or didn't exist, False on error
    """
    try:
        if path.exists():
            if not any(path.iterdir()):  # Check if empty
                path.rmdir()
            else:
                logging.warning(f"Directory not empty, not removing: {path}")
        return True
    except Exception as e:
        logging.error(f"Failed to clean directory {path}: {e}")
        return False

def get_relative_path(file_path: Path, base_path: Path) -> Path | None:
    """Get relative path from base path.
    
    Args:
        file_path: Full path to file
        base_path: Base path to make relative to
        
    Returns:
        Relative path or None if not relative to base_path
    """
    try:
        return file_path.relative_to(base_path)
    except ValueError:
        logging.error(f"Path {file_path} is not relative to {base_path}")
        return None 