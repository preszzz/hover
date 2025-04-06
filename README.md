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
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Organization

Place your drone audio datasets in the `data/raw/` directory with the following structure: 