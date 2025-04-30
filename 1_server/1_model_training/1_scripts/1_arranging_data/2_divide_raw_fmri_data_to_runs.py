############################################################################
# AUTHORS: Najemeddine Abdennour
# EMAIL: nabdennour@bcbl.eu
# COPYRIGHT: Copyright (C) 2024-2025, pyDecNef
# URL: https://github.com/najemabdennour/pyDecNef
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
This script organizes raw FMRI data into separate runs within a decoder's training directory.

The script begins by defining paths for where the raw FMRI data is stored and where it should be organized. It uses a test mode flag to adjust these paths if needed.

Steps:

1. Determine the main folder containing raw FMRI data.
2. Identify today's directories in this main folder.
3. Extract unique run IDs from filenames.
4. For each run, create a destination folder in the decoder's training directory.
5. Copy corresponding files into their respective run folders.

The script ensures that:

- Raw data is only moved if it matches the current run ID.
- Each run has its own subdirectory under 'raw/func'.
- The most recent directories are prioritized for organization.

Note: This script assumes that filenames contain a numeric sequence indicating the run ID after the volume ID numeric sequence separated by the underscore(e.g., ''001_000007_000001.dcm' corresponds to the first scan in run 7').
"""

import os
from datetime import datetime
import shutil

# Define the main folder path where MRI scanner dumps scans
recorded_data_main_folder = "/firmm/"

# Toggle for test mode to alter data paths.
test_mode = True
if test_mode:
    recorded_data_main_folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir,os.pardir)),"2.data","recorded_data")

# Print confirmation of the main data folder path
print("Main folder for recorded data:", recorded_data_main_folder)

# Get the list of directories in the main folder, excluding hidden files
data_folders = [os.path.isdir(i) for i in os.listdir(recorded_data_main_folder) if not i.startswith(".")]
# Check if there are any subdirectories in the main folder
if any(data_folders):
    if test_mode:
        recorded_directories_list_today = data_folders 

     # Create a list of valid directories today's date
    today_date_str = datetime.today().strftime("%Y%m%d")
    recorded_directories_list_today = [i for i in data_folders if i.startswith(today_date_str)] # filter the directories by the ones created today

    # Convert directory names to full paths
    recorded_data_directories_list_paths = [os.path.join(recorded_data_main_folder,i) for i in recorded_directories_list_today]

    # Sort directories by creation time
    recorded_data_directories_list_paths.sort(key=lambda x: os.path.getmtime(x)) # sort by the most recently created files/folders
    

    # Determine the most recent raw data folder.
    try:
        raw_training_data_path = recorded_data_directories_list_paths[-1] # choose the raw training data folder path based on the last created
    except IndexError:
        print(f"No new folders created today. Using fallback path:{recorded_data_main_folder}")
        raw_training_data_path = recorded_data_main_folder  # Fallback to main folder

# fallback to the main folder as the container of the raw training data if it does not contain  any folders
else:
    raw_training_data_path = recorded_data_main_folder

# Print confirmation of the chosen raw data path.
print(f"Chosen raw training path to extract runs from: {raw_training_data_path}")

# Extract all files from the raw data folder (excluding hidden files).
raw_data_list = [file for file in  os.listdir(raw_training_data_path) if not file.startswith(".")]

# Identify unique run IDs from filenames.
run_ids = [file.split("_")[1] for file in os.listdir(raw_data_list)]
unique_runs = sorted(list(set(run_ids)))

print("The number of runs found:",len(unique_runs),"| Runs:",unique_runs )

# Define the path to the decoder's training folder.
decoder_training_folder_path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir,"2.data"))
print("Decoder training data folder path:",decoder_training_folder_path)

# Organize files into runs within the decoder's raw/func directory.
for i, pattern in enumerate(unique_runs):
    dest_path = os.path.join(decoder_training_folder_path,"raw","func",f"run_{i}")

    if not os.path.exists(dest_path):
        # Create the destination folder and copy files.
        os.makedirs(dest_path)
        folders_exits = False
        print(f"Moving the run files to the decoder training directory for: run_{i}")
        for file in os.listdir(raw_data_list):
            if pattern in file.split("_")[1]:
                shutil.copy(src=os.path.join(raw_training_data_path,file),dst= os.path.join(dest_path,file))
    else:
        print(f"The run_{i} folder already exists..No action taken")
        folders_exits= True
if not folders_exits:
    print("FMRI data successfully organized into run folders.")