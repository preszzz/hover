"""
Command-line real-time drone detection using a trained AST model with microphone input.
This version doesn't require matplotlib and can run in headless environments.
"""

import os
import logging
import time
import numpy as np
import torch
import sounddevice as sd
from queue import Queue
import threading
import argparse
from datetime import datetime

# Import from project modules
import config
from models.transformer_model import build_transformer_model, get_feature_extractor
from utils.hardware import get_device, list_audio_devices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("drone_detection.log")
    ]
)

# --- Configuration ---
# Audio settings
SAMPLE_RATE = config.TARGET_SAMPLE_RATE  # 16kHz for AST
CHUNK_DURATION = config.CHUNK_LENGTH_MS / 1000  # in seconds (1.0s)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
DETECTION_THRESHOLD = 0.6  # Default confidence threshold for positive detection

# Model settings
MODEL_LOAD_DIR = os.path.join(config.ROOT_DIR, 'output_models')
MODEL_FILENAME = config.CHECKPOINT_FILENAME  # "ast_best_model"

# Device setup
DEVICE = get_device()

# Global variables
audio_queue = Queue()
prediction_history = []
is_running = True
detection_count = 0
start_time = None
save_detections = False
detections_dir = "detection_clips"


def load_model():
    """Load the trained AST model."""
    model_path = os.path.join(MODEL_LOAD_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    model = build_transformer_model(num_classes=2, model_checkpoint=config.MODEL_CHECKPOINT)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logging.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

    model.to(DEVICE)
    model.eval()
    return model


def audio_callback(indata, frames, time, status):
    """Callback for sounddevice to capture audio chunks."""
    if status:
        logging.warning(f"Audio callback status: {status}")
    
    # Get the audio data as float32 and put it in the queue
    audio_chunk = indata.copy().astype(np.float32).squeeze()
    
    # Normalize audio to [-1, 1] range if not already
    if np.max(np.abs(audio_chunk)) > 1.0:
        audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
        
    audio_queue.put(audio_chunk)


def save_audio_clip(audio_data, confidence):
    """Save detected drone audio for later analysis."""
    if not os.path.exists(detections_dir):
        os.makedirs(detections_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{detections_dir}/drone_detected_{timestamp}_{confidence:.2f}.npy"
    
    try:
        np.save(filename, audio_data)
        logging.info(f"Saved detection audio to {filename}")
    except Exception as e:
        logging.error(f"Failed to save audio clip: {e}")


def process_audio_thread(model, feature_extractor, threshold, save_clips=False):
    """Process audio chunks and make predictions in a separate thread."""
    global is_running, detection_count, start_time, save_detections
    save_detections = save_clips
    
    # Store start time for runtime calculation
    start_time = time.time()
    
    logging.info(f"Starting audio processing thread (detection threshold: {threshold:.2f})")
    
    # For calculating average level
    level_history = []
    
    while is_running:
        if not audio_queue.empty():
            # Get audio chunk from queue
            audio_chunk = audio_queue.get()
            
            # Calculate audio level (RMS)
            audio_level = np.sqrt(np.mean(np.square(audio_chunk)))
            level_history.append(audio_level)
            if len(level_history) > 50:
                level_history.pop(0)
            
            # Skip processing if chunk is too short
            if len(audio_chunk) < CHUNK_SAMPLES * 0.8:
                continue
                
            # Ensure correct length (pad or truncate)
            if len(audio_chunk) < CHUNK_SAMPLES:
                audio_chunk = np.pad(audio_chunk, (0, CHUNK_SAMPLES - len(audio_chunk)))
            elif len(audio_chunk) > CHUNK_SAMPLES:
                audio_chunk = audio_chunk[:CHUNK_SAMPLES]
            
            # Process audio with AST feature extractor
            inputs = feature_extractor(
                audio_chunk, 
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            )
            
            # Move to the correct device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get prediction and confidence
                pred_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0, pred_class].item()
                
                # Add to history
                prediction_history.append((pred_class, confidence))
                if len(prediction_history) > 100:
                    prediction_history.pop(0)
                
                # Count detections
                if pred_class == 1 and confidence >= threshold:  # 1 = drone
                    detection_count += 1
                    
                    # Calculate running stats
                    elapsed_time = time.time() - start_time
                    hours = int(elapsed_time // 3600)
                    minutes = int((elapsed_time % 3600) // 60)
                    seconds = int(elapsed_time % 60)
                    
                    # Format detection alert with ASCII art attention
                    alert = (
                        "\n" + "!" * 50 + "\n" +
                        f"DRONE DETECTED! #{detection_count}\n" +
                        f"Confidence: {confidence:.4f}\n" +
                        f"Timestamp: {hours:02d}:{minutes:02d}:{seconds:02d} (runtime)\n" +
                        f"Audio level: {audio_level:.4f} RMS\n" +
                        "!" * 50 + "\n"
                    )
                    logging.info(alert)
                    
                    # Save audio clip if enabled
                    if save_detections:
                        save_audio_clip(audio_chunk, confidence)
                
                # Log periodic status (every 60 chunks = ~60 seconds)
                if len(prediction_history) % 60 == 0:
                    avg_level = np.mean(level_history) if level_history else 0
                    elapsed_time = time.time() - start_time
                    hours = int(elapsed_time // 3600)
                    minutes = int((elapsed_time % 3600) // 60)
                    
                    status = (
                        f"\n--- STATUS UPDATE ---\n" +
                        f"Runtime: {hours:02d}h:{minutes:02d}m\n" +
                        f"Detections: {detection_count}\n" +
                        f"Current: {'DRONE' if pred_class == 1 else 'No drone'} " +
                        f"(confidence: {confidence:.2f})\n" +
                        f"Avg Audio Level: {avg_level:.4f} RMS\n" +
                        f"Detection threshold: {threshold:.2f}\n" +
                        "--------------------"
                    )
                    logging.info(status)
        
        time.sleep(0.05)  # Small sleep to prevent CPU overuse


def main():
    """Main function to run the CLI live drone detection system."""
    global is_running
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Real-time drone detection from microphone")
    parser.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD,
                       help="Confidence threshold for drone detection (0.0-1.0)")
    parser.add_argument("--list-devices", action="store_true",
                       help="List available audio input devices and exit")
    parser.add_argument("--device", type=int, default=None,
                       help="Audio input device ID to use")
    parser.add_argument("--save-clips", action="store_true",
                       help="Save audio clips of positive detections")
    parser.add_argument("--duration", type=int, default=0,
                       help="Duration to run in minutes (0 = unlimited)")
    
    args = parser.parse_args()
    
    # Just list devices and exit if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    # Set detection threshold
    threshold = max(0.0, min(1.0, args.threshold))  # Clamp between 0 and 1
    
    try:
        # Load model and feature extractor
        logging.info("Loading AST model and feature extractor...")
        model = load_model()
        feature_extractor = get_feature_extractor(config.MODEL_CHECKPOINT)
        
        # Start audio processing thread
        processing_thread = threading.Thread(
            target=process_audio_thread, 
            args=(model, feature_extractor, threshold, args.save_clips)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        # Calculate end time if duration is set
        end_time = None
        if args.duration > 0:
            end_time = time.time() + (args.duration * 60)
            logging.info(f"Will run for {args.duration} minutes")
        
        # Print welcome banner
        print("\n" + "=" * 50)
        print(" DRONE DETECTION MONITORING SYSTEM ")
        print("=" * 50)
        print(" Using external microphone for continuous detection")
        print(f" Detection threshold: {threshold:.2f}")
        if args.save_clips:
            print(f" Saving detection clips to: {detections_dir}/")
        if end_time:
            print(f" Will run for {args.duration} minutes")
        print(" Press Ctrl+C to stop")
        print("=" * 50 + "\n")
        
        # Start audio stream
        logging.info("Starting audio stream...")
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE,
                           blocksize=CHUNK_SAMPLES, device=args.device):
            
            # Keep running until duration expires or Ctrl+C
            if end_time:
                while time.time() < end_time and is_running:
                    time.sleep(1)
                logging.info("Scheduled duration completed")
            else:
                # Run indefinitely until Ctrl+C
                while is_running:
                    time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Stopping by user request (Ctrl+C)")
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)
    finally:
        # Clean up
        is_running = False
        if 'processing_thread' in locals() and processing_thread.is_alive():
            processing_thread.join(timeout=1.0)
        
        # Calculate and display final statistics
        if start_time:
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            
            # Calculate detections per hour
            if elapsed_time > 0:
                detections_per_hour = (detection_count / elapsed_time) * 3600
            else:
                detections_per_hour = 0
                
            summary = (
                "\n" + "#" * 50 + "\n" +
                "# SESSION SUMMARY\n" +
                f"# Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}\n" +
                f"# Total drone detections: {detection_count}\n" +
                f"# Detection rate: {detections_per_hour:.2f} per hour\n" +
                "#" * 50
            )
            logging.info(summary)


if __name__ == "__main__":
    main() 