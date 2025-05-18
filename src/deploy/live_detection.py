"""
Real-time drone detection using a trained AST model with microphone input.
"""

import logging
import time
import numpy as np
import torch
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from queue import Queue
import threading
from transformers import ASTForAudioClassification

# Import from project modules
import config
from models.transformer_model import get_feature_extractor
from utils.hardware import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device setup
DEVICE = get_device()

# Audio processing constants from config
SAMPLE_RATE = config.TARGET_SAMPLE_RATE
CHUNK_SAMPLES = config.CHUNK_LENGTH_SAMPLES
DETECTION_THRESHOLD = 0.7 # Default threshold, tune as needed

# Global variables
audio_queue = Queue()
prediction_history = []
is_running = True
detection_count = 0
current_prediction = None
current_confidence = 0.0

def audio_callback(indata, frames, time, status):
    """Callback for sounddevice to capture audio chunks."""
    if status:
        logging.warning(f"Audio callback status: {status}")
    
    # Get the audio data as float32 and put it in the queue
    audio_chunk = indata.copy().squeeze()
    
    audio_queue.put(audio_chunk)


def process_audio_thread(model, feature_extractor):
    """Process audio chunks and make predictions in a separate thread."""
    global is_running, current_prediction, current_confidence, detection_count
    
    logging.info("Starting audio processing thread")
    
    while is_running:
        if not audio_queue.empty():
            # Get audio chunk from queue
            audio_chunk = audio_queue.get()
            
            # Skip processing if chunk is too short
            if len(audio_chunk) < CHUNK_SAMPLES * 0.8:  # Allow some flexibility
                continue
                
            # Ensure correct length (pad or truncate)
            if len(audio_chunk) < CHUNK_SAMPLES:
                audio_chunk = np.pad(audio_chunk, (0, CHUNK_SAMPLES - len(audio_chunk)))
            elif len(audio_chunk) > CHUNK_SAMPLES:
                audio_chunk = audio_chunk[:CHUNK_SAMPLES]
            
            # Process audio with AST feature extractor
            processed_inputs = feature_extractor(
                audio_chunk, 
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            )
            
            # Move the necessary tensor to the correct device
            input_values = processed_inputs.input_values.to(DEVICE)
            
            # Make prediction
            with torch.no_grad():
                # Pass the tensor directly
                outputs = model(input_values) 
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get prediction and confidence
                pred_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0, pred_class].item()
                
                # Update global variables
                current_prediction = pred_class
                current_confidence = confidence
                
                # Add to history
                prediction_history.append((pred_class, confidence))
                if len(prediction_history) > 50:  # Keep only recent history
                    prediction_history.pop(0)
                
                # Count detections
                if pred_class == 1 and confidence >= DETECTION_THRESHOLD:  # 1 = drone
                    detection_count += 1
                    logging.info(f"DRONE DETECTED! Confidence: {confidence:.4f}")
                
                # Log periodic status
                if len(prediction_history) % 10 == 0:
                    logging.info(f"Current prediction: {'DRONE' if pred_class == 1 else 'No drone'}, "
                                f"Confidence: {confidence:.4f}, "
                                f"Total detections: {detection_count}")
        
        time.sleep(0.05)  # Small sleep to prevent CPU overuse


def update_plot(frame, ax_audio, ax_pred, line_audio, bar_pred):
    """Update function for the matplotlib animation."""
    global current_prediction, current_confidence
    
    # Get the latest audio chunk if available
    if not audio_queue.empty():
        audio_data = audio_queue.queue[-1]  # Peek at the last item
        line_audio.set_ydata(audio_data)
    
    # Update prediction bar
    if current_prediction is not None:
        bar_colors = ['blue', 'red']  # blue for no drone, red for drone
        
        # Remove old bars and create new ones
        ax_pred.clear()
        ax_pred.set_xlim(-0.5, 1.5)
        ax_pred.set_ylim(0, 1)
        ax_pred.set_xticks([0, 1])
        ax_pred.set_xticklabels(['No Drone', 'Drone'])
        ax_pred.set_ylabel('Confidence')
        ax_pred.set_title(f"Prediction: {'DRONE DETECTED!' if current_prediction == 1 else 'No drone'}")
        
        # Create the bars
        bars = ax_pred.bar([0, 1], 
                           [1 - current_confidence if current_prediction == 1 else current_confidence, 
                            current_confidence if current_prediction == 1 else 1 - current_confidence],
                           color=[bar_colors[0], bar_colors[1]])
        
        # Highlight the predicted class
        bars[current_prediction].set_alpha(1.0)
        bars[1 - current_prediction].set_alpha(0.3)
        
        # Add text for the confidence value
        for i, bar in enumerate(bars):
            if i == current_prediction:
                ax_pred.text(i, bar.get_height() / 2, f"{bar.get_height():.2f}", 
                             ha='center', va='center', color='white', fontweight='bold')
    
    return line_audio, 


def main():
    """Main function to run the live drone detection system."""
    global is_running
    
    try:
        # Load model and feature extractor
        logging.info("Loading AST model and feature extractor...")
        try:
            # Load the fine-tuned model directly
            model = ASTForAudioClassification.from_pretrained(
                config.MODEL_HUB_ID,
                cache_dir=config.CACHE_DIR
            )
            model.to(DEVICE)
            model.eval()
            feature_extractor = get_feature_extractor(config.MODEL_HUB_ID)
            logging.info("Model and feature extractor loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model from {config.MODEL_HUB_ID}: {e}")
            raise
        
        # Start audio processing thread
        processing_thread = threading.Thread(
            target=process_audio_thread, 
            args=(model, feature_extractor)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        # Set up the visualization
        fig, (ax_audio, ax_pred) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Audio waveform subplot
        ax_audio.set_ylim(-1, 1)
        ax_audio.set_xlim(0, CHUNK_SAMPLES)
        ax_audio.set_ylabel("Amplitude")
        ax_audio.set_title("Audio Waveform")
        line_audio, = ax_audio.plot(np.zeros(CHUNK_SAMPLES))
        
        # Prediction subplot
        ax_pred.set_xlim(-0.5, 1.5)
        ax_pred.set_ylim(0, 1)
        ax_pred.set_xticks([0, 1])
        ax_pred.set_xticklabels(['No Drone', 'Drone'])
        ax_pred.set_ylabel('Confidence')
        ax_pred.set_title("Prediction: Waiting...")
        bar_pred = ax_pred.bar([0, 1], [0, 0], color=['blue', 'red'])
        
        # Create animation for real-time updates
        ani = FuncAnimation(fig, update_plot, fargs=(ax_audio, ax_pred, line_audio, bar_pred),
                            interval=100, blit=False)
        
        # Start audio stream
        logging.info("Starting audio stream. Press Ctrl+C to stop.")
        with sd.InputStream(
            callback=audio_callback, 
            channels=1, 
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SAMPLES, 
            latency='low',
            dtype='float32'
            ):
            plt.show()
            
    except KeyboardInterrupt:
        logging.info("Stopping by user request (Ctrl+C)")
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)
    finally:
        # Clean up
        is_running = False
        if 'processing_thread' in locals() and processing_thread.is_alive():
            processing_thread.join(timeout=1.0)
        logging.info(f"Session summary: {detection_count} drone detections")

if __name__ == "__main__":
    main() 