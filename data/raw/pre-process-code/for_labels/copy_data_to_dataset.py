import json
import shutil
import os

def get_folder_name():
    with open("../file_with_folder_name.json", "r") as file:
        data = json.load(file)
        return data["folder_name"]


# Get the folder name
original_folder_name = get_folder_name()
# print(original_folder_name); exit()
# Define the source and destination paths
all_data = '../'+ original_folder_name + '/data'
destination = '../../FINISHED_V7'

# Process the folder name (remove '../') just before copying
processed_folder_name = original_folder_name.replace('../', '')
final_destination = os.path.join(destination, processed_folder_name)

# Copy the all_data folder to the destination with the modified name
shutil.copytree(all_data, final_destination)