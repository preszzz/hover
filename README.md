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

### Dataset 
1. **SOUND-BASED DRONE FAULT CLASSIFICATION USING MULTI-TASK LEARNING (Version 1)**  
   29th International Congress on Sound and Vibration (ICSV29), Wonjun Yi, Jung-Woo Choi, & Jae-Woo Lee. (2023). Prague. [Zenodo](https://doi.org/10.5281/zenodo.7779574)
   
2. **DREGON**  
   Audio-Based Search and Rescue with a Drone: Highlights from the IEEE Signal Processing Cup 2019 Student Competition. Antoine Deleforge, Diego Di Carlo, Martin Strauss, Romain Serizel, & Lucio Marcenaro. (2019). IEEE Signal Processing Magazine, 36(5), 138-144. Institute of Electrical and Electronics Engineers. [Kaggle](https://www.kaggle.com/datasets/awsaf49/ieee-signal-processing-cup-2019-dataset)

3. **Audio Based Drone Detection and Identification using Deep Learning**  
   Sara A Al-Emadi, Abdulla K Al-Ali, Abdulaziz Al-Ali, Amr Mohamed. [GitHub](https://github.com/saraalemadi/DroneAudioDataset/tree/master)

4. **DronePrint**  
   Harini Kolamunna, Thilini Dahanayake, Junye Li, Suranga Seneviratne, Kanchana Thilakaratne, Albert Y. Zomaya, Aruna Seneviratne. [GitHub](https://github.com/DronePrint/DronePrint/tree/master)

5. **Drone Detection and Classification using Machine Learning and Sensor Fusion**  
   Svanström F. (2020). [GitHub](https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/master)

6. **DroneNoise Database**  
   Carlos Ramos-Romero, Nathan Green, César Asensio and Antonio J Torija Martinez. [Figshare](https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411)

7. **ESC: Dataset for Environmental Sound Classification**  
   Piczak, Karol J. [GitHub](https://github.com/karolpiczak/ESC-50)

8. **drone-audio-detection**  
   [GitHub](https://github.com/BowonY/drone-audio-detection/tree/develop)