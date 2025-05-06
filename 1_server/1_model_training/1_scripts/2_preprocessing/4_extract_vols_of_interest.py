#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 This script extracts volumes of interest from functional MRI data based on:
     - Behavioral data (including trial onset times and target classifications)
     - Functional data alignment to reference space using ANI/AFNI tools
 
 The script performs the following steps:
     1. Loads functional and behavioral data
     2. Matches each run's functional data with its corresponding behavioral file
     3. Creates a time series representation of the functional data
     4. Extracts heatup volumes (first few TRs) as baseline for normalization
     5. Identifies time windows of interest based on trial onset times and HRF peaks
     6. Extracts and saves:
         - All volume data
         - Baseline volume data (after removing heatup)
         - Volumes of interest (based on experiment timing)
 
 The extracted volumes are saved in a directory structure matching the original data organization.
"""
# IMPORT BASIC DEPENDENCIES
from pathlib import Path
import pandas as pd
from nilearn.image import load_img, index_img
import os
# RELEVANT VARIABLES FOR EXTRACTING VOLS OF INTEREST
n_heatup_vols = 5
TR = 2000 # in miliseconds if using opensesame
onset_hrf_peak = 4000 # in miliseconds
offset_hrf_peak = 6000 # in miliseconds
trial_onset_time_column = 'trial_onset_time' # trial onset time column name in the behavioral data file
target_column ='concept' # target class column name in the behavioral data file
label_class_0 = 'non_living' # target class 0 label in the behavioral data file
label_class_1 = 'living' # target class 1 label in the behavioral data file
trial_idx_column = 'trial_idx' # trial idx column name in the behavioral data file

# SET FILE STRUCTURE
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )
preprocessed_dir =os.path.join(exp_dir,'2.data', 'preprocessed')
behav_dir = Path(os.path.join(exp_dir,'2.data','raw','behav'))
func_dir = Path(os.path.join(preprocessed_dir,'func'))
vols_of_interest_dir = os.path.join(preprocessed_dir,'vols_of_interest')
os.makedirs(vols_of_interest_dir,exist_ok=True)

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
    df_run = pd.read_csv(behav_file, usecols=[trial_onset_time_column, target_column])
    df_run[trial_onset_time_column] -= (TR * n_heatup_vols)
    df_run[trial_idx_column] = df_run.index
    df_run.drop(df_run.tail(1).index, inplace = True) # Remove last row as it was only used to track run's total duration

    # Extract each trial living or non.living category
    targets = []
    #for value in df_run['selected_class'].values:
    for value in df_run[target_column].values:
        try:
            if (label_class_0 in value):
                targets.append(0)
            elif (label_class_1 in value):
                targets.append(1)
            else:
                targets.append(2)
        except:
            targets.append(None)
    df_run['targets_category'] = targets

    # Get baseline vols
    baseline_vols_idxs = []
    for vol_idx, time in enumerate(new_timeseries):
        if time <= df_run[trial_onset_time_column].iloc[0]:
            baseline_vols_idxs.append(vol_idx)
    # Stablish intervals of interest based on trials_onsets
    df_run['hrf_peak_onset'] = df_run[trial_onset_time_column] + onset_hrf_peak
    df_run['hrf_peak_offset'] = df_run[trial_onset_time_column] + offset_hrf_peak

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
                vols_trial_idx.append(row[trial_idx_column])
                vols_trial_category.append(row['targets_category'])
                vols_trial_onset.append(row[trial_onset_time_column])
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
    nii_data.to_filename(os.path.join(vols_of_interest_dir, func_name + '_allvols.nii.gz'))
    # Extract baseline vols
    baseline_vols = index_img(nii_data, baseline_vols_idxs)
    baseline_vols.to_filename(os.path.join(vols_of_interest_dir,func_name + '_baseline.nii.gz'))
    # Extract vols_of_interest
    vols_of_interest = index_img(nii_data, vols_idxs_of_interest) # Extract just vols of interest data from all vols
    vols_of_interest.to_filename(os.path.join(vols_of_interest_dir, func_name + '_vols_of_interest.nii.gz'))
    df_vols.to_csv(os.path.join(vols_of_interest_dir,func_name + '_vols_of_interest.csv'))
