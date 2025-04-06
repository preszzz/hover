import os
import shutil
from pydub import AudioSegment


def convert_all_mp3_to_wav(input_folder_path, output_folder_name="converted_war"):
    # Create the output folder if it does not exist
    output_folder_path = os.path.join(input_folder_path, output_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(input_folder_path):
        if filename.endswith(".mp3"):
            mp3_file_path = os.path.join(input_folder_path, filename)
            wav_file_path = os.path.join(output_folder_path, filename.replace(".mp3", ".wav"))

            # Load the MP3 file
            audio = AudioSegment.from_mp3(mp3_file_path)

            # Export as WAV file
            audio.export(wav_file_path, format="wav")
            print(f"Converted {filename} to WAV in {output_folder_path}")

        elif filename.endswith(".wav"):
            # Copy the WAV file to the output folder without modification
            wav_file_path = os.path.join(input_folder_path, filename)
            output_wav_file_path = os.path.join(output_folder_path, filename)
            shutil.copyfile(wav_file_path, output_wav_file_path)
            print(f"Copied {filename} to {output_folder_path} without modification")


import json


def get_folder_name():
    with open("./file_with_folder_name.json", "r") as file:
        data = json.load(file)
        return data["folder_name"]


folder_name = get_folder_name()
print(folder_name)
convert_all_mp3_to_wav(folder_name)
