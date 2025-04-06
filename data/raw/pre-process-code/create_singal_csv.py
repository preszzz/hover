import os
import wave
import numpy as np
import shutil
import json

def get_folder_name():
    with open("./file_with_folder_name.json", "r") as file:
        data = json.load(file)
        return data["folder_name"]


splitted_dir = get_folder_name()+'/splitted_dir'

all_data = get_folder_name()+'/data'

print("start")

# Function to create directories and save first channel signal data to signal.csv
def create_directories_and_save_signal(src_root, dest_root):
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith('.wav'):
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_file_path, src_root)
                dest_file_path = os.path.join(dest_root, relative_path)

                # Create the directory if it does not exist
                dest_dir_path = os.path.dirname(dest_file_path)
                os.makedirs(dest_dir_path, exist_ok=True)
                # print(f'Created directory {dest_dir_path}')

                # Read the .wav file and extract the first channel data
                with wave.open(src_file_path, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()

                    # Read the frames and convert to numpy array
                    frames = wav_file.readframes(n_frames)
                    data = np.frombuffer(frames, dtype=np.int16)
                    data = data.reshape(-1, n_channels)

                    # Extract the first channel
                    c0 = data[:, 0]

                # Check if the signal data is smaller than (16000, 1)
                if c0.shape[0] < 16000:
                    print(f'Signal data is smaller than (16000, 1) for {src_file_path}')
                    # Delete the created directory
                    shutil.rmtree(dest_dir_path)
                    print(f'Deleted directory {dest_dir_path}')
                else:
                    # Save the first channel data to signal.csv
                    signal_csv_path = os.path.join(dest_dir_path, 'signal.csv')
                    np.savetxt(signal_csv_path, c0, delimiter=',', fmt='%d')
                    print(f'Created signal.csv at {signal_csv_path}')

                    # Check if signal.csv is full of zeros
                    if np.all(c0 == 0):
                        print(f'signal.csv is full of zeros for {src_file_path}')
                        shutil.rmtree(dest_dir_path)
                        print(f'Deleted directory {dest_dir_path}')
                        continue

                    # Check if the data can cause a RuntimeWarning or error in other code
                    try:
                        audio_data = c0.astype(np.float32)
                        audio_data /= np.max(np.abs(audio_data))
                        if not np.isfinite(audio_data).all():
                            raise ValueError("Audio buffer is not finite everywhere")
                    except Exception as e:
                        print(f'Error encountered with {src_file_path}: {e}')
                        shutil.rmtree(dest_dir_path)
                        print(f'Deleted directory {dest_dir_path}')


# Call the function
create_directories_and_save_signal(splitted_dir, all_data)
