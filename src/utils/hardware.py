import logging
import torch
import sounddevice as sd

# --- Device Setup ---
def get_device():
    """
    Determine the best available device for PyTorch operations.
    
    Returns:
        torch.device: The device to use (CUDA GPU, MPS for Apple Silicon, or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        # For Apple Silicon Macs
        device = torch.device("mps")
        logging.info("Using Apple Silicon MPS acceleration")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for computation")
    
    return device


def list_audio_devices():
    """
    Lists all available audio input devices.
    Useful for selecting the correct external microphone.
    
    Returns:
        None: Prints device information to console
    """
    try:
        print("\nAvailable Audio Input Devices:")
        print("-" * 30)
        
        for device in sd.query_devices():
            # Only show input devices or devices with input channels
            if device['max_input_channels'] > 0:
                print(f"\nInput Device Name: {device['name']}")
                print(f"Input Device ID: {device['index']}")
                print(f"Max Input Channels: {device['max_input_channels']}")
                print(f"Default Sample Rate: {device['default_samplerate']}")
                
        # Show default input device
        default_input = sd.query_devices(kind='input')
        print(f"\nDefault input device: {default_input['name']}")
        
    except Exception as e:
        print(f"Error listing audio devices: {e}")
