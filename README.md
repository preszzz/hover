# Acoustic Drone Detection System

This project implements an acoustic system for detecting and classifying drones based on their sound signatures.

## Project Overview

Drones produce distinctive sounds due to their motors, propellers, and aerodynamics. This project leverages machine learning and audio signal processing to identify and classify drones from audio recordings.

## Setup and Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd drone_detection_capstone
   ```

2. Create a virtual environment (recommended):
   ```bash
   # use uv to create a virtual environment and install dependencies
   uv sync
   # macOS
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate
   ```

## Data Organization

Place your drone audio datasets in the `data/raw/` directory. Each distinct dataset should reside in its own subdirectory:

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

The preprocessing pipeline converts raw audio files into features suitable for model training. It is located in `src/preprocessing/` and consists of the following steps, orchestrated by `main_preprocess.py`:

1.  **Convert to WAV (`step_1_convert_to_wav.py`):** Converts various audio formats (e.g., MP3) to WAV format and standardizes existing WAV files (PCM16). Output goes to `data/interim/wav/`.
2.  **Resample (`step_2_resample.py`):** Resamples audio to a target sample rate (defined in `config.py`, default 16kHz). Skips resampling if the audio is already at the target rate. Output goes to `data/interim/resampled/`.
3.  **Split (`step_3_split.py`):** Splits the resampled audio into fixed-length chunks (defined in `config.py`, default 1 second). Skips files shorter than the chunk length. Output goes directly into `data/processed/`, maintaining the relative directory structure from the source dataset (e.g., `data/processed/<DatasetName>/<Subdirs>/<FileName>/<FileName_chunk_N>/`).
4.  **Extract Features (`step_4_extract_features.py`):** For each 1-second chunk:
    *   Validates the chunk (checks length, silence).
    *   Saves the raw audio signal as `signal.csv` (int16 format).
    *   Normalizes the signal and extracts Mel-Frequency Cepstral Coefficients (MFCCs) as defined in `config.py` (default: 40 MFCCs over 32 frames).
    *   Saves the MFCC array as `mfcc.npy`.
    *   Invalid segments are deleted.
5.  **Create Label Files (`step_5_create_label_files.py`):**
    *   Reads labeling rules defined in `src/preprocessing/label_mapping.yaml`.
    *   Determines the label (e.g., 'drone', 'nodrone', 'similar_car') for each processed segment based on its original source path/filename and the defined rules.
    *   Writes the determined label into a `label.txt` file within each segment's directory.

### Running the Pipeline

1.  **Configure Labeling:**
    *   **Crucially, edit `src/preprocessing/label_mapping.yaml`**. Define rules for each dataset directory in `data/raw/` to correctly map files/subdirs to labels (e.g., 'drone', 'nodrone'). See the comments and examples within the file.
2.  **Ensure Dependencies:** Make sure all required Python packages are installed (e.g., via `uv sync`). You might also need `ffmpeg` installed on your system if processing MP3 files (for `librosa`'s fallback mechanism).
3.  **Execute:** Run the main script from the project root directory:
    ```bash
    uv src/preprocessing/main_preprocess.py
    ```

The pipeline will generate intermediate files in `data/interim/` and the final processed segments (containing `signal.csv`, `mfcc.npy`, `label.txt`) directly under `data/processed/`. These directories are ignored by git by default (see `.gitignore`).

## Model Training (To Be Added)

...

## Evaluation (To Be Added)

...