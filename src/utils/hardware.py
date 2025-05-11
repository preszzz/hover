import torch
import logging

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
        import sounddevice as sd
        devices = sd.query_devices()
        
        print("\nAvailable Audio Input Devices:")
        print("-" * 50)
        
        for i, device in enumerate(devices):
            # Only show input devices or devices with input channels
            if device['max_input_channels'] > 0:
                print(f"Device {i}: {device['name']}")
                print(f"  Max Input Channels: {device['max_input_channels']}")
                print(f"  Default Sample Rate: {device['default_samplerate']}")
                if 'hostapi' in device:
                    print(f"  Host API: {device['hostapi']}")
                print()
                
        # Show default input device
        default_input = sd.query_devices(kind='input')
        print(f"Default input device: {default_input['name']}")
        
    except ImportError:
        print("sounddevice module not installed. Install with 'uv pip install sounddevice'")
    except Exception as e:
        print(f"Error listing audio devices: {e}")


if __name__ == "__main__":
    # When run directly, display hardware information
    device = get_device()
    print(f"Using device: {device}")
    
    # List available audio input devices
    list_audio_devices()