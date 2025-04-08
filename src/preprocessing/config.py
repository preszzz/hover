"""Configuration settings for the audio preprocessing pipeline."""

import os

# --- Paths ---
# Assumes the script is run from the project root or uses relative paths correctly
# You might need to adjust these based on your execution context
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Project root
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Specific subdirectories for pipeline steps
RESAMPLED_DIR = os.path.join(INTERIM_DATA_DIR, 'resampled')
INTERIM_SPLIT_DIR = os.path.join(INTERIM_DATA_DIR, 'split_with_meta')

# --- Audio Processing Parameters ---
TARGET_SAMPLE_RATE = 16000  # Hz (16kHz)
CHUNK_LENGTH_MS = 1000      # milliseconds (1 second)
CHUNK_LENGTH_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_LENGTH_MS / 1000)  # Number of samples in one chunk

# --- MFCC Parameters (Based on reference project) ---
N_MFCC = 40          # Number of MFCC coefficients
N_FFT = 2048         # FFT window size
HOP_LENGTH = 512     # Hop length for STFT
FMAX = 8000          # Maximum frequency for MEL filters (Hz)
NORMALIZATION_DB = -20.0 # dB level for normalization reference

# --- Output File Names ---
SIGNAL_FILENAME = 'signal.csv'   # Raw signal data as CSV
MFCC_FILENAME = 'mfcc.npy'       # MFCC features as numpy array

# --- Misc ---
NUM_WORKERS = os.cpu_count()  # For potential parallel processing (optional) 