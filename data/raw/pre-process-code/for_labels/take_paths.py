import os
import csv
import json

def find_mfcc_csv(root_dir):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'mfcc.csv' in filenames:
            paths.append(dirpath)

    csv_filename = f'{os.path.basename(root_dir)}_paths.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Directory Path'])
        for path in paths:
            writer.writerow([path])

def get_folder_name():
    with open("../file_with_folder_name.json", "r") as file:
        data = json.load(file)
        return data["folder_name"]

folder_name = '../' + get_folder_name()

# Replace 'your_root_directory' with the path to the root directory you want to search
find_mfcc_csv(folder_name)
