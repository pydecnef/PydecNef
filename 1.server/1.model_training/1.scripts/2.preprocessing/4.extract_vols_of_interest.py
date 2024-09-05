#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# IMPORT BASIC DEPENDENCIES
from pathlib import Path
import pandas as pd
from nilearn.image import load_img, index_img

# RELEVANT VARIABLES FOR EXTRACTING VOLS OF INTEREST
n_heatup_vols = 5
TR = 2000 # in miliseconds if using opensesame
onset_hrf_peak = 4000 # in miliseconds
offset_hrf_peak = 6000 # in miliseconds

# SET FILE STRUCTURE
###################################################
### najem addons
import os
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )    #Path().absolute() 
raw_vols_dir = os.path.join(exp_dir, '2.data/raw/func/')
raw_vols_dir = Path(raw_vols_dir)
preprocessed_dir =Path(os.path.join(exp_dir,'2.data', 'preprocessed'))
behav_dir = Path(os.path.join(exp_dir,'2.data/raw/behav'))
###
###################################################
func_dir = preprocessed_dir / 'func'
vols_of_interest_dir = preprocessed_dir / 'vols_of_interest'
vols_of_interest_dir.mkdir(exist_ok = True, parents = True)

# MATCH FUNC TO BEHAV
func_files = sorted([file for file in func_dir.glob('**/*.nii.gz')]) # List all runs preprocessed functional data
behav_files = sorted([csv for csv in behav_dir.glob('*.csv') if not str(csv).split("/")[-1].startswith(".")]) # List all behavioral files
exp_data = zip(func_files, behav_files)

# LABEL PREPROCESSED FUNCTIONAL VOLS WITH BEHAVIORAL DATA
for func_file, behav_file in exp_data:
    
    # Get run name
    func_parent = func_file.parent
    func_name = func_parent.name
    
    # Set paths as strings   
    func_file = str(func_file)
    behav_file = str(behav_file)
    print(func_file, behav_file)
    
    # Remove heatup vols from MRI data
    nii_data = load_img(func_file)
    nii_data = index_img(nii_data, slice(n_heatup_vols, None))
    
    # Set first vol after heatup as new session onset (i.e., time 0) and create a new timeseries for functional data (increase time for each volume based on specified TR)
    onset_time = 0 # ms
    new_timeseries = []
    for i in range(nii_data.shape[-1]):
        new_timeseries.append(onset_time)
        onset_time += TR
    
    # Extract corresponding heatup vols times from behavioral data
    #df_run = pd.read_csv(behav_file, usecols=['trial_onset_time', 'selected_class'])
    try:
        df_run = pd.read_csv(behav_file, usecols=['trial_onset_time', 'concept'])
    except:
        df_run = pd.read_csv(behav_file, usecols=['trial_onset', 'concept'])
    try:
        df_run['trial_onset_time'] -= (TR * n_heatup_vols)
    except:
        df_run['trial_onset'] -= (TR * n_heatup_vols)
    df_run['trial_idx'] = df_run.index
    df_run.drop(df_run.tail(1).index, inplace = True) # Remove last row as it was only used to track run's total duration

    # Extract each trial living or non.living category
    targets = []
    #for value in df_run['selected_class'].values:
    for value in df_run['concept'].values:
        try:
            #if ('right' in value):
            if ('non_living' in value):
                targets.append(0)
            elif ('living' in value):
                targets.append(1)
            else:
                targets.append(2)
        except:
            targets.append('nan')
    df_run['targets_category'] = targets
    
    # Get baseline vols
    baseline_vols_idxs = []
    for vol_idx, time in enumerate(new_timeseries):
        try:
            if time <= df_run['trial_onset_time'].iloc[0]:
                baseline_vols_idxs.append(vol_idx)
        except:
            if time <= df_run['trial_onset'].iloc[0]:
                baseline_vols_idxs.append(vol_idx)
    
    # Stablish intervals of interest based on trials_onsets
    try:
        df_run['hrf_peak_onset'] = df_run['trial_onset_time'] + onset_hrf_peak
        df_run['hrf_peak_offset'] = df_run['trial_onset_time'] + offset_hrf_peak
    except:
        df_run['hrf_peak_onset'] = df_run['trial_onset'] + onset_hrf_peak
        df_run['hrf_peak_offset'] = df_run['trial_onset'] + offset_hrf_peak
    
    # Assign vols to trials based on intervals of interest
    vols_idxs_of_interest = []
    vols_of_interest_times = []
    vols_trial_idx = []
    vols_trial_category = []
    vols_trial_item = []
    vols_trial_onset = []
    vols_trial_hrf_peak_onset = []
    vols_trial_hrf_peak_offset = []
    
    for vol_idx, time in enumerate(new_timeseries):
        for row in df_run.iterrows():
            row = row[1]
            if row['hrf_peak_onset'] <= time <= row['hrf_peak_offset']:
                vols_idxs_of_interest.append(vol_idx)
                vols_of_interest_times.append(time)
                vols_trial_idx.append(row['trial_idx'])
                vols_trial_category.append(row['targets_category'])
                try:
                    vols_trial_onset.append(row['trial_onset_time'])
                except:
                    vols_trial_onset.append(row['trial_onset'])
                vols_trial_hrf_peak_onset.append(row['hrf_peak_onset'])
                vols_trial_hrf_peak_offset.append(row['hrf_peak_offset'])
                
    df_vols = {'vol_idx': vols_idxs_of_interest,
               'vol_time': vols_of_interest_times,
               'trial_idx': vols_trial_idx,
               'target_category': vols_trial_category,
               'trial_onset': vols_trial_onset,
               'hrf_peak_onset': vols_trial_hrf_peak_onset,
               'hrf_peak_offset': vols_trial_hrf_peak_offset,
               'run': func_name,
               }
    
    df_vols = pd.DataFrame(df_vols)
    
    # Save vols without heatup vols
    nii_data.to_filename(str(vols_of_interest_dir / (func_name + '_allvols.nii.gz')))
    
    # Extract baseline vols
    baseline_vols = index_img(nii_data, baseline_vols_idxs)
    baseline_vols.to_filename(str(vols_of_interest_dir / (func_name + '_baseline.nii.gz')))
    
    # Extract vols_of_interest
    vols_of_interest = index_img(nii_data, vols_idxs_of_interest) # Extract just vols of interest data from all vols
    vols_of_interest.to_filename(str(vols_of_interest_dir / (func_name + '_vols_of_interest.nii.gz')))
    df_vols.to_csv(str(vols_of_interest_dir / (func_name + '_vols_of_interest.csv')))
