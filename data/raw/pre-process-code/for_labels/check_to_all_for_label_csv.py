import os

def check_label_csv(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        if not dirs:  # Check if there are no subdirectories
            if 'label.csv' in files:
                pass
                # print(f"'label.csv' found in: {subdir}")
            else:
                print(f"No 'label.csv' in: {subdir}")


def get_folder_name():
    with open("../file_with_folder_name.json", "r") as file:
        import json
        data = json.load(file)
        return data["folder_name"]

folder_name = '../' + get_folder_name()+'/data'
# print(folder_name) ; exit()
check_label_csv(folder_name)
