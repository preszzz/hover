# Acoustic Drone Detection System

An acoustic system for detecting and classifying drones based on their sound signatures using machine learning and audio processing.

## Setup

```bash
# Clone and enter the project directory
git clone <repository-url>
cd drone_detection_capstone

# Install dependencies
uv sync
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

## Data Organization

Place audio datasets in `data/raw/`, with each dataset in its own subdirectory:

```
data/raw/
├── DatasetName1_Source/
│   ├── drone_audio/...
│   └── nodrone_audio/...
├── DatasetName2_Source/
│   ├── UAV_Sounds/...
│   └── Ambient_Noise/...
└── ... (other datasets)
```

The pipeline relies on the structure within these dataset directories (and potentially filenames) to determine labels.

## Preprocessing Pipeline

The pipeline in `src/preprocessing/` processes raw audio into features for model training in two streamlined steps:

1. **Convert and Resample** (`step_1_resample.py`):
   * Converts audio to WAV format (PCM16)
   * Resamples to target rate (default: 16kHz)
   * Validates format and sample rate
   * Outputs to `data/interim/`

2. **Process and Label** (`step_2_process.py`):
   * Splits audio into fixed-length chunks (default: 1 second)
   * Validates chunks (length, silence)
   * Extracts features (raw signals and MFCCs)
   * Applies labels based on `label_mapping.yaml` rules
   * Organizes output into `data/processed/DatasetName/Label/Chunk/`
   * Shows progress with completion percentage

The pipeline provides clear progress tracking, showing the total number of files to process and current completion percentage.

### Running the Pipeline

1. **Configure Labels**: Edit `src/preprocessing/label_mapping.yaml` with your dataset rules
2. **Run**: Execute from project root:
   ```bash
   uv run src/preprocessing/main_preprocess.py
   ```

The final output in `data/processed/` will contain organized chunks with:
- Raw signal data (`.csv`)
- MFCC features (`.npy`)
- Label information (`.txt`)

## Required Dependencies

- Python 3.x
- librosa
- soundfile
- numpy
- PyYAML
- ffmpeg (optional: for MP3 support via librosa)