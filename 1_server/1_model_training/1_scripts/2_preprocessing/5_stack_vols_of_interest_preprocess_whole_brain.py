#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script preprocesses functional MRI volumes of interest by:
    1. Loading and transforming NIFTI images into numpy arrays.
    2. Normalizing the volume data using z-score.
    3. Stacking the normalized volumes across multiple runs.

The script assumes that:
    - The input files (baseline, allvols, and vols_of_interest) exist in the specified directory structure.
    - The NIFTI images are compatible with the NiftiMasker tool for transformation.

Directory Structure:
    - exp_dir: Path to this script's directory
        - 2.data/
            - preprocessed/
                - example_func/
                - vols_of_interest/
                    - *_baseline.nii.gz
                    - *_allvols.nii.gz
                    - *_vols_of_interest.csv
                - stacked_vols_of_interest/

Key Steps:
    1. Load and transform NIFTI images into numpy arrays.
    2. Normalize the data using z-score.
    3. Stack volumes of interest across runs.
    4. Save processed data in a structured directory.


Settings:
    detrend: Whether to remove linear trends from the data.

Setup Directories:
    Creates necessary subdirectories if they don't exist.
    
Functions:
    - zscore_func: Normalizes the volume data by subtracting the mean and dividing by the standard deviation.
    - load_data: Loads baseline, allvols, and vols_of_interest files.

Processing Loop:
    Iterates over each run's data, extracting and processing volumes of interest.
    
Saving Data:
    Saves stacked volumes as NIFTI image and numpy arrays, along with mean/standard deviation for reference.
"""
# IMPORT BASIC DEPENDENCIES
from pathlib import Path
from nilearn.image import load_img, new_img_like
from nilearn.maskers import NiftiMasker
import pandas as pd
import numpy as np
from nilearn.signal import _detrend
import os
# SETUP VARIABLES
detrend = True

# SET FILE STRUCTURE
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )   
preprocessed_dir =os.path.join(exp_dir, '2_data','preprocessed')
example_func_dir = Path(os.path.join(preprocessed_dir,'example_func'))
vols_of_interest_dir = Path(os.path.join(preprocessed_dir,'vols_of_interest'))
stacked_vols_of_interest_dir = os.path.join(preprocessed_dir,'stacked_vols_of_interest')
os.makedirs(stacked_vols_of_interest_dir,exist_ok=True)

baseline_vols = sorted([str(run) for run in vols_of_interest_dir.glob('*_baseline.nii.gz')]) # Retrieve paths of baselines for all runs
all_vols = sorted([str(run) for run in vols_of_interest_dir.glob('*_allvols.nii.gz')])
vols_of_interest_csv = sorted([str(run) for run in vols_of_interest_dir.glob('*_vols_of_interest.csv')])
runs_data = list(zip(baseline_vols, all_vols, vols_of_interest_csv))

# USE NIFTIMASKER TO CONVERT NILEARN IMGS TO FLATTEN NUMPY ARRAYS AND UNCONVERT TO NILEARN IMG FORMAT (THIS ALLOWS TO MAP AND TRACK EACH VOXEL IN THE 3D SPACE TO THE 2D SPACE)
example_func = str(next(example_func_dir.glob('example_func_deoblique_brain.nii'))) # Example func file
example_func = load_img(example_func)
dummy_mask = np.ones(example_func.shape[0:3])  # To just take the whole brain without masking anything
dummy_mask = new_img_like(example_func, dummy_mask, copy_header = True)
    
nifti_masker = NiftiMasker(mask_img = dummy_mask,
                           smoothing_fwhm = None,
                           standardize = False, # Normalize data using just vols of interest whole brain information
                           detrend = False,
                           )

def zscore_func(array):
    def zscore(array, mean_array, std_array):
        array = array - mean_array
        array /= std_array
        return array
    
    mean_array = array.mean(axis = 0)
    std_array = array.std(axis = 0)
    std_array[std_array < np.finfo(np.float64).eps] = 1. # Avoid numerical problems
    
    zscored_vols = zscore(array, mean_array, std_array)

    return zscored_vols, mean_array, std_array

vols_of_interest = []

for baseline_vols, all_vols, vols_of_interest_csv in runs_data:
    print(baseline_vols, all_vols, vols_of_interest_csv)
    func_name = Path(baseline_vols).name.split('_baseline')[0]
    
    baseline_vols = load_img(baseline_vols)
    all_vols = load_img(all_vols)
    vols_of_interest_csv = pd.read_csv(vols_of_interest_csv)
    
    baseline_vols = nifti_masker.fit_transform(baseline_vols) # Convert baseline to flatten numpy array
    all_vols = nifti_masker.fit_transform(all_vols)
    
    vols_of_interest_idxs = vols_of_interest_csv.vol_idx.values
    
    all_vols = _detrend(all_vols, inplace = False, type = 'linear', n_batches = 10)
    
    vols_of_interest.append(all_vols[vols_of_interest_idxs])

# STACK FMRI DATA    
vols_of_interest = np.vstack(vols_of_interest)
zscored_vols, mean_array, std_array = zscore_func(vols_of_interest)
vols_of_interest = nifti_masker.inverse_transform(zscored_vols) # Return zscored array to a 4D image


  
# STACK BEHAV DATA
behav_perception = sorted([str(run_behav) for run_behav in vols_of_interest_dir.glob('*.csv')])
behav_perception = [pd.read_csv(run_behav) for run_behav in behav_perception]
stacked_behav_perception = pd.concat(behav_perception, ignore_index = True)

# SAVE STACKED DATA
vols_of_interest.to_filename(os.path.join(stacked_vols_of_interest_dir,'detrended_zscored_stacked_vols_of_interest.nii.gz'))
np.save(os.path.join(stacked_vols_of_interest_dir,'zscoring_mean_array.npy'), mean_array)
np.save(os.path.join(stacked_vols_of_interest_dir,'zscoring_std_array.npy'), std_array)

# SAVE STACKED BEHAV DATA
stacked_behav_perception.to_csv(os.path.join(stacked_vols_of_interest_dir,'detrended_zscored_stacked_vols_of_interest_labels.csv'))
