# Live Drone Detection Testing

This guide explains how to use the external microphone to continuously monitor and test for drone sounds using our trained AST model.

## Prerequisites

1. A trained AST model (should be in `output_models/ast_best_model`)
2. An external microphone connected to your system
3. Python dependencies installed

## Setup Instructions

1. **Install required dependencies**

```bash
uv pip install sounddevice matplotlib numpy torch
```

2. **Set the microphone as default input device (optional)**

For macOS:
- Go to System Preferences > Sound > Input
- Select your external microphone

For Windows:
- Right-click the sound icon in taskbar > Sound settings
- Set your external microphone as default input device

3. **Adjust the detection threshold** (if needed)

Open `src/live_detection.py` and modify the `DETECTION_THRESHOLD` value (default: 0.6):
- Higher values (e.g., 0.8) make the system less sensitive but more precise
- Lower values (e.g., 0.4) make it more sensitive but may increase false positives

## Running the Live Detection

1. From the project root directory, run:

```bash
uv run src/live_detection.py
```

2. The system will:
   - Load the trained AST model
   - Start capturing audio from your microphone
   - Process audio in 1-second chunks (matching the training data)
   - Show a real-time visualization with:
     - Audio waveform display
     - Prediction confidence for drone/no-drone classes
   - Log detections to the console

3. Press `Ctrl+C` to stop the program

## Monitoring Features

- **Real-time waveform**: Shows the audio being captured
- **Confidence bars**: Visual representation of model confidence
- **Console logs**: Records each detection event and periodic status
- **Session summary**: Displays total detections when you exit

## Troubleshooting

1. **Model not found error**
   - Ensure you've trained the model and it exists in the output_models directory
   - Check if the model filename in `config.py` matches your actual model file

2. **Audio input issues**
   - Check if your microphone is properly connected and selected
   - Try running a simple audio recording test with another program
   - Adjust input volume if the signal is too low

3. **Performance issues**
   - If the system is lagging, try closing other applications
   - For slower machines, you may need to adjust the animation interval in `live_detection.py`

## Extending the System

If you want to enhance the live detection system:

1. **Add audio recording capabilities**
   - Save audio clips of positive detections for later analysis

2. **Implement notification systems**
   - Add email/SMS alerts for positive detections
   - Integrate with external systems or databases

3. **Improve visualization**
   - Add a spectrogram view of the audio
   - Create a historical view of detection confidence over time 