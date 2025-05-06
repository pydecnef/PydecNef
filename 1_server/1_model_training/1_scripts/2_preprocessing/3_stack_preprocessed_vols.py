#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script concatenates all raw functional MRI (fMRI) volumes from multiple runs into a single volume per run.

The script processes data stored in the following directory structure:
    - The root directory is determined by the location of this script.
    - Inside, there's an 'exp_dir' which points to the experiment directory.
    - Within 'exp_dir', there's a '2.data/preprocessed' subdirectory containing the processed data.
    - Each run (folder) under 'preprocessed/func' contains raw image files ('*.nii') that need to be concatenated.

Steps:
    1. Determine the root and experiment directories.
    2. Identify all functional directories under 'preprocessed/func'.
    3. For each run directory, collect all .nii files.
    4. Sort these files by their numeric index in the filename.
    5. Concatenate the sorted images into a single volume per run.
    6. Delete the individual files after concatenation to save space.

The code begins by setting up the necessary imports and directories, then iterates over each run, processes its volumes, stacks them, saves the result, and cleans up by removing temporary files.
"""
from pathlib import Path
from nilearn.image import concat_imgs

# Set file structure
import os
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )    #Path().absolute()
preprocessed_dir = os.path.join(exp_dir,'2.data', 'preprocessed')
func_dir = Path(os.path.join(preprocessed_dir,'func'))
# STACK ALL PREPROCESSED RAW VOLS BY RUN
for folder in func_dir.iterdir():
    if folder.is_dir():
        # Collect all .nii files in the current run
        vols = [vol_file for vol_file in folder.glob('*.nii')]
        # Sort volumes by their index from filename
        sorted_vols = sorted(vols, key = lambda vol_file: int(vol_file.name.split('_')[2])) 
        # Convert paths to strings for concatenation
        sorted_vols_str = [str(vol_file) for vol_file in sorted_vols] # Convert Pathlib format routes to str
        print(sorted_vols_str)
        print('\n')
        # Concatenate images into a single volume
        stacked_run = concat_imgs(sorted_vols_str) 
        # Create output file path
        stacked_file =  os.path.join(folder ,folder.name + '.nii.gz')
        # Save the concatenated volume as a single .nii.gz file corresponding to each run folder within preprocessed/func_dir
        stacked_run.to_filename(stacked_file)
        # Remove individual files to free memory
        for vol_file in sorted_vols:
            vol_file.unlink()