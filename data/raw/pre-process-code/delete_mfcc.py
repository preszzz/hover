import os

# Specify the root directory
root_dir = "../FINISHED_V6"

# Walk through the directory
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == "mfcc.csv":
            file_path = os.path.join(subdir, file)
            os.remove(file_path)
            print(f"Deleted {file_path}")
