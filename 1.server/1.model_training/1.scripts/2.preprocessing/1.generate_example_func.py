#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NOTES
# Before running load dcm2niix and AFNI first in the cluster
# Change brainextraction.inputs.clfrac to improve brain extraction

# IMPORT BASIC DEPENDENCIES
from pathlib import Path
from nipype.interfaces import afni as afni
from nilearn.image import load_img, concat_imgs, index_img
import pydicom
from nibabel.nicom import dicomreaders
import nibabel as nib
import numpy as np
import subprocess

# SET FILE STRUCTURE
###################################################
### najem addons
import os
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )    #Path().absolute() 
raw_vols_dir = os.path.join(exp_dir, '2.data/raw/func/')
raw_vols_dir = Path(raw_vols_dir)
preprocessed_dir =Path(os.path.join(exp_dir,,'2.data','preprocessed'))
###
###################################################
preprocessed_dir.mkdir(exist_ok = True, parents = True)
example_func_dir = preprocessed_dir / 'example_func'
example_func_dir.mkdir(exist_ok = True, parents = True)

# DEOBLIQUE EXAMPLE_FUNC PIPELINE

# Load vol DICOM, convert to Nifti and store the result in output_dir
ref_vol = raw_vols_dir / 'run_1/001_000005_000015.dcm' # First baseline volume from run-1
####################################################
### najem addons
import os
run_dir = os.path.join(raw_vols_dir,[i for i in os.listdir(raw_vols_dir) if not i.startswith(".")][-1] ) 
ref_vol = os.path.join(run_dir, os.listdir(run_dir)[0])
print("the used file for the example:",ref_vol)
###################################################
vol_name = 'example_func'
subprocess.run([f'dcm2niix -z n -f {vol_name} -o {example_func_dir} -s y {ref_vol}'], shell = True)
nifti_file = example_func_dir / (vol_name + '.nii') # To save each vol as .nii instead to .nii.gz to load faster

# Deoblique converted Nifti file
deoblique_vol = afni.Warp() # Use AFNI 3dWarp command
deoblique_vol.inputs.in_file = nifti_file
deoblique_vol.inputs.deoblique = True # Deoblique Nifti files
deoblique_vol.inputs.num_threads = 4 # Set number of threads for processing
deoblique_vol.inputs.outputtype = 'NIFTI'
ref_vol_deoblique_file = example_func_dir / (vol_name + '_deoblique.nii')
deoblique_vol.inputs.out_file = ref_vol_deoblique_file
deoblique_vol.run()

# Perform brain extraction to improve session to session registration of functional data
brainextraction = afni.Automask() # Use AFNI Automask command
brainextraction.inputs.in_file = ref_vol_deoblique_file
brainextraction.inputs.erode = 1 # Erode the mask inwards to avoid skull and tissue fragments. Check this parameter for each subject based on brain extraction performance during training session.
brainextraction.inputs.clfrac = 0.5 # Sets the clip level fraction (0.1 - 0.9). By default 0.5. The larger the restrictive brain extraction is.
brainextraction.inputs.num_threads = 4 # Set number of threads for processing
brainextraction.inputs.outputtype = 'NIFTI'
brain_file = example_func_dir / (vol_name + '_deoblique_brain.nii')
brainmask_file = example_func_dir / (vol_name + '_deoblique_brainmask.nii')
brainextraction.inputs.brain_file = brain_file # Just brain's data
brainextraction.inputs.out_file = brainmask_file # Brain binarized mask
brainextraction.run()
