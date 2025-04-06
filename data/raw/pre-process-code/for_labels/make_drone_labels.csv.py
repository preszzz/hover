import pandas as pd
import os

# Read 'all_drones.csv'
all_paths_df = pd.read_csv('all_drones.csv', header=None)

# Iterate through each row in the DataFrame
for index, row in all_paths_df.iterrows():
    path = row[0]

    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # Create the label.csv file with 'not_drone' in [0][0]
    labels_df = pd.DataFrame([['drone']])

    # Define the full path for the new label.csv file
    label_csv_path = os.path.join(path, 'label.csv')

    # Save the DataFrame to a CSV file
    labels_df.to_csv(label_csv_path, index=False, header=False)

    # Print confirmation
    print(f"'label.csv' created at {label_csv_path} with 'drone' in [0][0]")
