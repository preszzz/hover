# Acoustic Drone Detection System

An acoustic system for detecting and classifying drones based on their sound signatures, primarily using the Audio Spectrogram Transformer (AST) model.

## Setup

```bash
# Clone and enter the project directory (if you haven't already)
git clone https://github.com/preszzz/hover.git
cd hover
cp .env.example .env
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

## Overview

This project uses a pre-trained Audio Spectrogram Transformer (AST) model from Hugging Face for binary classification of audio signals (drone vs. non-drone). Key features include:

*   **AST Model:** Leverages "MIT/ast-finetuned-audioset-10-10-0.4593" as a base. The model was fine-tuned on a substantial dataset (approximately 585k training samples, 32k validation samples, and 32k test samples) for the drone detection task.
*   **Fine-tuned Model on Hugging Face Hub:** The fine-tuned model is available on the Hugging Face Hub: [Drone Acoustic Detection](https://huggingface.co/preszzz/drone-acoustic-detection)
*   **On-the-Fly Feature Extraction:** Uses `ASTFeatureExtractor` to convert raw audio (resampled to 16kHz) into spectrograms during data loading. No pre-computation and storage of spectrograms is required.
*   **Hugging Face `datasets`:** Manages data loading and transformations. Supports loading from Hugging Face Hub or local audio folders.
*   **PyTorch Framework:** Model training and evaluation are implemented in PyTorch.
*   **Hyperparameter Tuning:** A simplified Optuna-based script (`hyperparameter/tune.py`) is provided for finding optimal hyperparameters. Results (best parameters) are logged to the console.
*   **Configuration:** Core settings are managed in `src/config.py`.

## Data Handling

The system primarily uses Hugging Face `datasets` for data input.

*   **Hugging Face Hub:** You can load datasets directly from the Hub (e.g., `load_dataset("your_username/your_dataset_name")`).
*   **Local Audio Files:** Use `load_dataset("audiofolder", data_dir="data/raw/your_audio_data")`. Your audio files should be organized into subdirectories where each subdirectory name corresponds to a label (e.g., `drone`, `non_drone`).
    ```
    data/raw/your_audio_data/
    ├── drone/
    │   ├── drone_audio_001.wav
    │   └── ...
    └── non_drone/
        ├── ambient_sound_001.wav
        └── ...
    ```
The AST feature extractor requires audio to be at a **16kHz sampling rate**. The data loading script (`src/feature_engineering/feature_loader.py`) handles casting the audio to this rate.

## Core Workflow

1.  **Configuration (`src/config.py`):**
    *   Review and adjust base parameters like model checkpoint, batch size, learning rate, number of epochs if needed.
    *   Define `ID2LABEL` and `LABEL2ID` mappings.

2.  **Hyperparameter Tuning (Optional, Recommended):**
    *   The script `hyperparameter/tune.py` uses Optuna to find optimal hyperparameters.
    *   It runs in-memory and prints the best parameters to the console.
    *   Modify the script to define search spaces if defaults are not suitable.
    ```bash
    uv run hyperparameter/tune.py
    ```
    *   Manually update `src/config.py` or a separate optimal config file with the best parameters found.

3.  **Training (`src/training/train.py`):**
    *   This script trains the AST model using settings from `src/config.py` (or your optimal config).
    *   It loads data, performs on-the-fly feature extraction, and saves the best model checkpoint (based on validation performance) to `trained_models_pytorch/`.
    ```bash
    uv run src/training/train.py
    ```

4.  **Evaluation (`src/training/evaluate.py`):**
    *   Evaluates the trained model (from `trained_models_pytorch/`) on a test set.
    *   Outputs metrics like accuracy, precision, recall, F1-score, and a classification report.
    *   Results can be found in `evaluation_results_pytorch/`.
    ```bash
    uv run python src/training/evaluate.py
    ```

## MFCC Preprocessing Pipeline

**(Note: The primary workflow for this project now uses the AST model with on-the-fly feature extraction. The following describes a legacy MFCC-based preprocessing pipeline which may be used for other models or auxiliary tasks. Its components are located in `src/preprocessing/`.)**

This older pipeline processes raw audio into MFCC features:

1.  **Convert and Resample** (`src/preprocessing/step_1_resample.py`): Converts to WAV, resamples to 16kHz.
2.  **Process and Label** (`src/preprocessing/step_2_process.py`): Chunks audio, extracts MFCCs, applies labels via `src/preprocessing/label_mapping.yaml`.

### Running the Legacy Pipeline:
1.  Configure `src/preprocessing/label_mapping.yaml`.
2.  Run: `uv run src/preprocessing/main_preprocess.py`
    Output: `data/processed/` with `.npy` features and `.txt` labels.
   - Raw signal data (`.npy`)
   - MFCC features (`.npy`)
   - Label information (`.txt`)

## Required Dependencies

*   Python 3.11
*   `uv` (for package management)
*   `torch`
*   `transformers`
*   `datasets`
*   `evaluate`
*   `optuna`
*   `scikit-learn`
*   `numpy`
*   `librosa`
*   `soundfile`

Installation is handled by `uv sync` based on `pyproject.toml`.

## Example Public Datasets (for context or use)

(This section lists publicly available drone audio datasets. You may need to adapt their structure or use Hugging Face `datasets` tools to load them.)

1.  **Audio Based Drone Detection and Identification using Deep Learning**
    *   Sara A Al-Emadi, Abdulla K Al-Ali, Abdulaziz Al-Ali, Amr Mohamed.
    *   [GitHub](https://github.com/saraalemadi/DroneAudioDataset/tree/master)

2.  **Drone Detection and Classification using Machine Learning and Sensor Fusion**
    *   Svanström F. (2020).
    *   [GitHub](https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/master)

3.  **DREGON**
    *   Audio-Based Search and Rescue with a Drone: Highlights from the IEEE Signal Processing Cup 2019 Student Competition. Antoine Deleforge, Diego Di Carlo, Martin Strauss, Romain Serizel, & Lucio Marcenaro. (2019). IEEE Signal Processing Magazine, 36(5), 138-144.
    *   [Kaggle](https://www.kaggle.com/datasets/awsaf49/ieee-signal-processing-cup-2019-dataset)

4.  **DronePrint**
    *   Harini Kolamunna, Thilini Dahanayake, Junye Li, Suranga Seneviratne, Kanchana Thilakaratne, Albert Y. Zomaya, Aruna Seneviratne.
    *   [GitHub](https://github.com/DronePrint/DronePrint/tree/master)

5.  **DroneNoise Database**
    *   Carlos Ramos-Romero, Nathan Green, César Asensio and Antonio J Torija Martinez.
    *   [Figshare](https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411)

6.  **ESC: Dataset for Environmental Sound Classification** (General environmental sounds, useful for non-drone class)
    *   Piczak, Karol J.
    *   [GitHub](https://github.com/karolpiczak/ESC-50)

7.  **drone-audio-detection**
    *   [GitHub](https://github.com/BowonY/drone-audio-detection/tree/develop)

8.  **SOUND-BASED DRONE FAULT CLASSIFICATION USING MULTI-TASK LEARNING (Version 1)**
    *   29th International Congress on Sound and Vibration (ICSV29), Wonjun Yi, Jung-Woo Choi, & Jae-Woo Lee. (2023).
    *   [Zenodo](https://doi.org/10.5281/zenodo.7779574)
