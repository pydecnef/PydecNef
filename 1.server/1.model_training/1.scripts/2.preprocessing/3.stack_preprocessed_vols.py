#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Ensure sorting line is actually correctly sorting volumes taking into account each volume file path

# IMPORT BASIC DEPENDENCIES
from pathlib import Path
from nilearn.image import concat_imgs

# SET FILE STRUCTURE
###################################################
### najem addons
import os
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )    #Path().absolute() 
preprocessed_dir =Path(os.path.join(exp_dir,'2.data', 'preprocessed'))
###
###################################################
func_dir = preprocessed_dir / 'func'

# STACK ALL PREPROCESSED RAW VOLS BY RUN
for folder in func_dir.iterdir():
    if folder.is_dir():
        vols = [vol_file for vol_file in folder.glob('*.nii')] # List all preprocessed vols of this run
        sorted_vols = sorted(vols, key = lambda vol_file: int(vol_file.name.split('_')[2])) # Ensure vols are sorted by their index
        sorted_vols_str = [str(vol_file) for vol_file in sorted_vols] # Convert Pathlib format routes to str
        print(sorted_vols_str)
        print('\n')
        stacked_run = concat_imgs(sorted_vols_str) # Then concat all vols in one 4D nifti file
        stacked_file = folder / (folder.name + '.nii.gz')
        stacked_run.to_filename(stacked_file) # Save stacked data as a single .nii.gz file corresponding to each run folder within preprocessed/func_dir
        for vol_file in sorted_vols:
            vol_file.unlink() # Delete individual .nii files in each run folder to free space
        
