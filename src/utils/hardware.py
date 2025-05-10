import torch
import logging

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