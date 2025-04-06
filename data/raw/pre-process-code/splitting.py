import os
from pydub import AudioSegment
import json

def get_folder_name():
    with open("./file_with_folder_name.json", "r") as file:
        data = json.load(file)
        return data["folder_name"]


resampled_dir = get_folder_name()+'/resampled_data'
splitted_dir = get_folder_name()+'/splitted_dir'

print(splitted_dir) ; print(resampled_dir)
print("start")

def split_wav(file_path, output_dir, chunk_length_ms=1000):
    audio = AudioSegment.from_wav(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    total_duration = len(audio)
    num_full_chunks = total_duration // chunk_length_ms

    for i in range(num_full_chunks):
        chunk = audio[i * chunk_length_ms:(i + 1) * chunk_length_ms]
        chunk_name = f"{file_name}_chunk_{i + 1}.wav"
        chunk_dir = os.path.join(output_dir, f"{file_name}_chunk_{i + 1}")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk.export(os.path.join(chunk_dir, chunk_name), format="wav")


def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                split_wav(file_path, output_subdir)


process_directory(resampled_dir, splitted_dir)
