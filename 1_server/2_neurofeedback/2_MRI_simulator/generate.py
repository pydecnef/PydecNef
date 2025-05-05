############################################################################
# AUTHORS: Pedro Margolles & David Soto
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2021-2022, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""

This script uses the example raw functional MRI data from a subject's session (i.e., DICOM files) to simulate 
 a MRI scanner by moving files from the 'real_data' directory to the 'outputs' directory.
This process is meant to test the experimental paradigm running in parallel 
 neurofeedback scripts as it was a real session

"""


import shutil
import time
from pathlib import Path
import os
import shutil
from colorama import init, Fore # For colouring outputs in the terminal
init()



#############################################################################################
# fMRI SIMULATION VARIABLES
#############################################################################################

TR = 2


#############################################################################################
# DIRECTORIES & DATA
#############################################################################################

# Get generate.py script directory
script_dir = Path(__file__).absolute().parent 

# Define an outputs directory to copy DICOM files from real_data directory as it was a real scanner
# this folder has to be present already there!!!
outputs_dir = Path(os.path.join(script_dir,'outputs'))

# Remove all files in outputs_dir (if there are)
if outputs_dir.exists(): 
    for file in outputs_dir.glob('*.*'): # Get all volumes in outputs folder
        file.unlink() # Remove each volume one by one

# Folder with fMRI DICOM files
real_data = Path(os.path.join(script_dir,'real_data'))
#############################################################################################
# DATA TRANSFER FROM REAL_DATA FOLDER TO OUTPUTS FOLDER
#############################################################################################
for volume in sorted(list(real_data.glob('*'))):
    print(Fore.YELLOW + f'\n[PROCESSING] Generating volume {volume.stem}...')
    time.sleep(TR) # Wait for a TR before generating the next volume
    shutil.copy(str(volume), str(outputs_dir))
    print(Fore.GREEN + '[OK]')
    

    
