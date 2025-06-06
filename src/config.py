"""Configuration settings for the pipeline."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Paths ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Project root
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

DATASET_NAME = os.getenv("HF_DATASET_ID")
CACHE_DIR = os.getenv("CACHE_DIR")

# --- Audio Processing Parameters ---
TARGET_SAMPLE_RATE = 16000  # Hz (16kHz)
CHUNK_LENGTH_MS = 1000      # milliseconds (1 second)
CHUNK_LENGTH_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_LENGTH_MS / 1000)  # Number of samples in one chunk

# --- MFCC Parameters ---
N_MFCC = 40                                     # Number of MFCC coefficients
N_FFT = 2048                                    # FFT window size
WIN_LENGTH = int(0.025 * TARGET_SAMPLE_RATE)    # Window length for STFT
HOP_LENGTH = 512                                # Hop length for STFT
N_MELS = 128                                    # Number of MEL filters
FMIN = 50                                       # Minimum frequency for MEL filters (Hz)
FMAX = int(TARGET_SAMPLE_RATE // 2)             # Maximum frequency for MEL filters (Hz)
NORMALIZATION_DB = -20.0                        # dB level for normalization reference

# --- Output File Names ---
SIGNAL_FILENAME = 'signal.npy'   # Raw signal data as numpy array
MFCC_FILENAME = 'mfcc.npy'       # MFCC features as numpy array
LABEL_FILENAME = 'label.txt'     # Label as text file

# --- Misc ---
NUM_WORKERS = os.cpu_count()

# --- Training Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-5

# --- Model Configuration ---
MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_SAVE_DIR = 'output_models'
MODEL_HUB_ID = os.getenv("HF_MODEL_ID")

# --- Hyperparameter Tuning ---
STUDY_NAME = "drone-audio-detection-05-17"
N_TRIALS = 10