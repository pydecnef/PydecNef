#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# IMPORT BASIC DEPENDENCIES
from pathlib import Path
from nilearn.image import concat_imgs, load_img, iter_img, new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask
from nilearn.signal import clean
import pandas as pd
import numpy as np
from nilearn.signal import _detrend
import time

# SETUP VARIABLES
detrend = True

# SET FILE STRUCTURE
exp_dir = Path().absolute().parent.parent
raw_vols_dir = exp_dir / '2.data/raw/func'
preprocessed_dir = exp_dir / '2.data/preprocessed/'
example_func_dir = preprocessed_dir / 'example_func'
preprocessed_func_dir = preprocessed_dir / 'func'
vols_of_interest_dir = preprocessed_dir / 'vols_of_interest'
rois_dir = preprocessed_dir / 'rois/'
stacked_vols_of_interest_dir = preprocessed_dir / 'stacked_vols_of_interest'
stacked_vols_of_interest_dir.mkdir(exist_ok = True, parents = True)
masks = ['bilateral']

baseline_vols = sorted([str(run) for run in vols_of_interest_dir.glob('*_baseline.nii.gz')]) # Retrieve paths of baselines for all runs
all_vols = sorted([str(run) for run in vols_of_interest_dir.glob('*_allvols.nii.gz')])
vols_of_interest_csv = sorted([str(run) for run in vols_of_interest_dir.glob('*_vols_of_interest.csv')])
runs_data = list(zip(baseline_vols, all_vols, vols_of_interest_csv))


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
    #mask_file = load_img(str(rois_dir / f'{mask}_mask_adapted.nii'))
    mask_file = load_img(str(rois_dir / f'full_mask.nii.gz'))
   # mask_file = load_img(str(rois_dir / f'lingual_native.nii.gz'))
    #mask_file = load_img(str(rois_dir / f'occcipitalfusiform_native.nii.gz'))
    #mask_file = load_img(str(rois_dir / f'tempoccipFusiform_native.nii.gz'))


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
        #zscored_vols = clean(signals = all_vols, standardize = 'zscore', detrend = False, ensure_finite = True)
        
        #np.save(str(vols_of_interest_dir / (func_name + f'_detrended_zscored_{mask}_allvols.npy')), all_vols, allow_pickle = True)
        
        vols_of_interest = all_vols[vols_of_interest_idxs]
        runs_vols_of_interest.append(vols_of_interest)
        
        #np.save(str(vols_of_interest_dir / (func_name + f'_detrended_zscored_{mask}.npy')), zscored_vols, allow_pickle = True)
      
    runs_vols_of_interest = np.vstack(runs_vols_of_interest)
    zscored_vols, mean_array, std_array = zscore_func(runs_vols_of_interest)
    np.save(str(stacked_vols_of_interest_dir / f'zscoring_mean_array_{mask}.npy'), mean_array)
    np.save(str(stacked_vols_of_interest_dir / f'zscoring_std_array_{mask}.npy'), std_array)
    
    stacked_perception = zscored_vols.copy()
      
    # STACK MRI DATA
    #perception_data = sorted([str(run) for run in vols_of_interest_dir.glob(f'*_detrended_zscored_{mask}.npy')]) # Retrieve paths to vols of interest of all runs
    #stacked_perception = []
    #for perception_array in perception_data:
    #    perception_array = np.load(perception_array)
    #    stacked_perception.append(perception_array)
    #stacked_perception = np.vstack(stacked_perception)
    
    # STACK BEHAV DATA
    behav_perception = sorted([str(run_behav) for run_behav in vols_of_interest_dir.glob(f'*.csv')])
    behav_perception = [pd.read_csv(run_behav) for run_behav in behav_perception]
    stacked_behav_perception = pd.concat(behav_perception, ignore_index = True)
    
    # SAVE STACKED DATA
    np.save(str(stacked_vols_of_interest_dir / f'perception_detrended_zscored_stacked_vols_of_interest_{mask}.npy'), stacked_perception)
    stacked_behav_perception.to_csv(str(stacked_vols_of_interest_dir / f'perception_detrended_zscored_stacked_vols_of_interest.csv'))
    
    
