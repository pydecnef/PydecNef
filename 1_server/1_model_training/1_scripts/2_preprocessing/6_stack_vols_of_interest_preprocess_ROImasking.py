#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# IMPORT BASIC DEPENDENCIES
from pathlib import Path
from nilearn.image import load_img
from nilearn.masking import apply_mask
import pandas as pd
import numpy as np
from nilearn.signal import _detrend
import os
# SETUP VARIABLES
detrend = True
# SET FILE STRUCTURE
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )   
preprocessed_dir =os.path.join(exp_dir, '2.data','preprocessed')
example_func_dir = Path(os.path.join(preprocessed_dir,'example_func'))
vols_of_interest_dir = Path(os.path.join(preprocessed_dir,'vols_of_interest'))
rois_dir = os.path.join(exp_dir,'2.data','rois_masks')

# list of mask names available in rois_mask folder used in the ROI masking process 
masks = os.listdir(rois_dir)
masks = [i.split(".")[0] for i in masks if not i.startswith(".") ]

baseline_vols = sorted([str(run) for run in vols_of_interest_dir.glob('*_baseline.nii.gz')]) # Retrieve paths of baselines for all runs
all_vols = sorted([str(run) for run in vols_of_interest_dir.glob('*_allvols.nii.gz')])
vols_of_interest_csv = sorted([str(run) for run in vols_of_interest_dir.glob('*_vols_of_interest.csv')])
runs_data = list(zip(baseline_vols, all_vols, vols_of_interest_csv))

# a function to calculate the z_score of volumes arrays
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


for mask in masks:
    stacked_vols_of_interest_dir = os.path.join(preprocessed_dir,f'masked_stacked_vols_of_interest_{mask}')
    os.makedirs(stacked_vols_of_interest_dir,exist_ok=True)
    print(f"\nMask used: {mask}\nData Saved in: {stacked_vols_of_interest_dir}\n")
    try:
        mask_file = load_img(os.path.join(rois_dir,f'{mask}.nii.gz'))
    except ValueError:
        mask_file = load_img(os.path.join(rois_dir,f'{mask}.nii'))
    runs_vols_of_interest = []
    for baseline_vols, all_vols, vols_of_interest_csv in runs_data:
        print(baseline_vols, all_vols, vols_of_interest_csv)
        func_name = Path(baseline_vols).name.split('_baseline')[0]
        
        baseline_vols = load_img(baseline_vols)
        all_vols = load_img(all_vols)
        vols_of_interest_csv = pd.read_csv(vols_of_interest_csv)
        
        baseline_vols = apply_mask(imgs = baseline_vols, mask_img = mask_file, smoothing_fwhm = None, ensure_finite = True) # Convert baseline to flatten numpy array
        all_vols = apply_mask(imgs = all_vols, mask_img = mask_file, smoothing_fwhm = None, ensure_finite = True)
        vols_of_interest_idxs = vols_of_interest_csv.vol_idx.values
        all_vols = _detrend(all_vols, inplace = False, type = 'linear', n_batches = 10)
        vols_of_interest = all_vols[vols_of_interest_idxs]
        runs_vols_of_interest.append(vols_of_interest)
      
    # STACK FMRI DATA
    runs_vols_of_interest = np.vstack(runs_vols_of_interest)
    zscored_vols, mean_array, std_array = zscore_func(runs_vols_of_interest)

    # STACK BEHAV DATA
    behav_data = sorted([str(run_behav) for run_behav in vols_of_interest_dir.glob(f'*.csv')])
    behav_data = [pd.read_csv(run_behav) for run_behav in behav_data]
    stacked_behav_perception = pd.concat(behav_data, ignore_index = True)
    
    # SAVE STACKED DATA
    np.save(os.path.join(stacked_vols_of_interest_dir, f'detrended_zscored_stacked_vols_of_interest_{mask}.npy'), zscored_vols)
    np.save(os.path.join(stacked_vols_of_interest_dir,f'zscoring_mean_array_{mask}.npy'), mean_array)
    np.save(os.path.join(stacked_vols_of_interest_dir, f'zscoring_std_array_{mask}.npy'), std_array)

    # SAVE STACKED BEHAV DATA
    stacked_behav_perception.to_csv(os.path.join(stacked_vols_of_interest_dir, f'detrended_zscored_stacked_vols_of_interest_labels.csv'))
    
    
