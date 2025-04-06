# Analysis of Preprocessing Pipeline

This document analyzes the audio preprocessing pipeline and suggests adaptable approaches for your capstone project.

## Confirmed Preprocessing Pipeline Diagram

The following diagram illustrates the specific steps outlined in the project's documentation for the stage:

```mermaid
graph TD
    A[Raw Audio Files (.wav, .mp3)] --> B{Convert MP3 to WAV};
    B --> C{Resample Audio (to 16kHz)};
    C --> D{Split Audio (into 1-sec segments)};
    D --> E{Extract MFCC Features};
    D --> F{Save Raw Segment Signal};
    E --> G[MFCC Data per Segment (e.g., mfcc.csv)];
    F --> H[Raw Signal Data per Segment (e.g., signal.csv, int16)];

    subgraph "Input Data"
        A
    end

    subgraph "Preprocessing Steps"
        B[convertion_mp3_wav.py]
        C[resampling.py]
        D[splitting.py]
        E[make_mfcc.py]
        F[create_signal_csv.py]
    end

    subgraph "Intermediate Outputs per Segment"
        G
        H
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:1px
    style H fill:#cff,stroke:#333,stroke-width:1px
```

**Explanation of Steps:**

1.  **Raw Audio Files**: Input audio data.
2.  **Convert MP3 to WAV (`convertion_mp3_wav.py`)**: Standardizes the audio format to WAV.
3.  **Resample Audio (`resampling.py`)**: Standardizes the sample rate, specifically to **16kHz** according to the README.
4.  **Split Audio (`splitting.py`)**: Breaks down the resampled audio into **1-second duration segments**.
5.  **Extract MFCC Features (`make_mfcc.py`)**: Calculates Mel-Frequency Cepstral Coefficients for each 1-second segment. The README indicates the CNN uses an input shape of `(40, 32, 1)`, suggesting **40 MFCC coefficients** are calculated over **32 time frames** for each 1-second segment. These are likely saved as a 2D array in a file like `mfcc.csv` for each segment.
6.  **Save Raw Segment Signal (`create_signal_csv.py`)**: Saves the raw waveform data of the 1-second segment (16000 samples) into a separate `signal.csv`, possibly as `int16` data type. *This step might be for analysis, specific model types, or debugging, but the primary feature for the documented CNN appears to be the MFCCs.*
7.  **Intermediate Outputs**: For each 1-second segment of the original audio, this process generates separate files (e.g., `mfcc.csv`, `signal.csv`).
8.  **(Later Consolidation - Implied by README)**: The README describes subsequent steps in `Create_Dataset_and_train` (`make_make_mfcc_labels.py`, `saveh5.py`) that likely gather these individual `mfcc.csv` files and labels, consolidating them into larger `.h5` files (e.g., `train_dataset.h5`) for efficient loading during model training.

## Highlighted Extracted Features

*   **Mel-Frequency Cepstral Coefficients (MFCCs)**:
    *   *Why*: Confirmed as the primary feature extracted for the CNN model described. Excellent for capturing timbral characteristics relevant to drone motor sounds.
    *   *Confirmed Components*: The script `make_mfcc.py` performs this extraction. The CNN input shape `(40, 32, 1)` strongly implies:
        *   **Number of Coefficients**: 40 MFCCs.
        *   **Temporal Information**: The features are preserved over 32 time frames within the 1-second segment, resulting in a `(40, 32)` array per segment, not just aggregated statistics like mean/variance *at this stage*.
    *   *Deltas*: While often used, the README and script names don't explicitly confirm if delta/delta-delta features are calculated by `make_mfcc.py` itself or handled later.

*   **Raw Signal Segment**:
    *   The `create_signal_csv.py` script saves the raw 16000 audio samples for each 1-second segment. It's unclear if this is directly used as a feature for the main model or serves another purpose (e.g., analysis, alternative models).

*   **Other Features (Not Confirmed in `pre-process-code`)**:
    *   The analysis of the `pre-process-code` directory and its corresponding README section **does not** show evidence of extracting Spectral Features (Centroid, Bandwidth, Contrast, Rolloff) or Temporal Features (ZCR, RMS) *within this specific set of scripts*.
    *   While these features *could* potentially be calculated in later stages (e.g., during the creation of the final `.h5` files) or used in different experiments within the larger project (`Compare/`), the core preprocessing described focuses solely on **MFCC sequences** and the raw signal segment.

## Adaptable Approaches for Your Capstone Project (Revised)

Your current pipeline (`audio_to_features.py`, `run_preprocessing.py`) provides a flexible way to extract *multiple* feature types (MFCCs, Spectral, Temporal, Mel stats) and aggregate them into a single feature vector per segment, saved in one consolidated CSV (`features_dataset.csv`). This is a valid and often effective approach, especially for models like Random Forests (used in `validate_features.py`) or other classifiers that expect flat feature vectors.

**Comparison and Recommendations:**

1.  **Feature Set**:
    *   *Reference Project (`pre-process-code`)*: Focuses primarily on generating **sequences of MFCCs** (e.g., 40 coefficients over 32 time frames) per 1-second segment, likely intended for a CNN architecture that processes 2D input.
    *   *Your Current Pipeline*: Extracts a **broader set of features** (MFCC stats, spectral stats, temporal stats, Mel stats) and **aggregates them** into a single vector per segment.
2.  **Output Format**:
    *   *Reference Project*: Creates individual files per segment (`mfcc.csv`, `signal.csv`) initially, then likely consolidates MFCC sequences into `.h5` files for training.
    *   *Your Current Pipeline*: Creates a single, large CSV file (`features_dataset.csv`) containing aggregated feature vectors for all segments.
3.  **Adaptation Strategy**:
    *   **Stick with Your Current Pipeline (Recommended for Now)**: Your existing approach is excellent for initial exploration, analysis (`dataset_analysis.py`), and validation with models like Random Forest (`validate_features.py`). The broader feature set might capture different aspects of the sound. Proceed with steps 4, 5, and 6 outlined previously using your current scripts.
    *   **Future: Target a CNN like the Reference?**: If you later decide to implement a CNN architecture *identical* to the one in the reference project (expecting `(40, 32, 1)` input), you would need to:
        *   Modify `audio_to_features.py`: Change `extract_features_from_audio` to return the raw MFCC sequence (e.g., a `(40, 32)` NumPy array) instead of aggregated statistics.
        *   Modify `process_dataset_to_csv`: Change how data is saved. Instead of appending rows to a CSV, you might save each segment's MFCC array as a separate `.npy` file or structure the data differently (e.g., within an HDF5 file) to be loaded efficiently by a deep learning framework like TensorFlow/Keras.
    *   **Hybrid Approach**: You could potentially train *different* models: one (like Random Forest) on your aggregated CSV features, and another (like a CNN) on MFCC sequences generated by a modified pipeline.

**In summary:** The reference project's documented preprocessing is specifically tailored to generate MFCC sequences for its CNN. Your current pipeline is more general, extracting aggregated statistics from a wider feature set, which is perfectly suitable for initial analysis and models like Random Forest. Continue with your current pipeline for now, and consider adapting it *only if* you specifically need to replicate the reference project's CNN input format later. 