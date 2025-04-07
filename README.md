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

The pipeline in `src/preprocessing/` processes raw audio into features for model training:

1. **Convert to WAV** (`step_1_convert_to_wav.py`): Standardizes audio to WAV format (PCM16)
2. **Resample** (`step_2_resample.py`): Converts to target sample rate (default: 16kHz)
3. **Split** (`step_3_split.py`): Divides into fixed-length chunks (default: 1 second)
4. **Extract Features** (`step_4_extract_features.py`): 
   * Validates chunks (length, silence)
   * Saves raw signals (`.csv`) and MFCCs (`.npy`) 
5. **Create Labels** (`step_5_create_label.py`):
   * Applies labeling rules from `label_mapping.yaml`
   * Organizes final output into `data/processed/DatasetName/Label/Chunk/`

### Running the Pipeline

1. **Configure Labels**: Edit `src/preprocessing/label_mapping.yaml` with your dataset rules
2. **Run**: Execute from project root:
   ```bash
   uv run src/preprocessing/main_preprocess.py
   ```

The final output in `data/processed/` will contain organized chunks with features (`.npy`, `.csv`) and labels (`label.txt`).

## Required Dependencies

- Python 3.x
- librosa
- soundfile
- numpy
- PyYAML
- ffmpeg (optional: for MP3 support via librosa)