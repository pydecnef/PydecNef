#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script performs data processing steps on functional MRI (fMRI) data.

The steps are as follows:
    1. Set up directory structure for the experiment.
    2. Identify and prepare the raw functional volume for processing.
    3. Convert the raw DICOM file to a NIFTI format using dcm2niix.
    4. Deoblique correct the NIFTI file using AFNI's 3dWarp tool.
    5. Extract the brain from the image using AFNI's Automask tool with specific parameters.

The script assumes that:
    - The raw DICOM files are available in a directory named 'raw' under '2.data'.
    - The processed data should be stored in a structure under 'preprocessed' directory, with example
functions stored in a subdirectory called 'example_func'.

The code is written to run on a Unix-based system. It uses AFNI and dcm2niix tools for processing.
"""
from nipype.interfaces import afni as afni
import subprocess
import os
# Set up the directory structure
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )    #Path().absolute()
raw_vols_dir = os.path.join(exp_dir, '2.data','raw','func')
preprocessed_dir =os.path.join(exp_dir,'2.data','preprocessed')
os.makedirs(preprocessed_dir,exist_ok=True)
example_func_dir = os.path.join(preprocessed_dir , 'example_func')
os.makedirs(example_func_dir,exist_ok=True)

# Step 2: Identify the raw functional volume
run_dir = os.path.join(raw_vols_dir,[i for i in os.listdir(raw_vols_dir) if not i.startswith(".")][-1] )
ref_vol = os.path.join(run_dir, os.listdir(run_dir)[0])

print("the used file for the example:",ref_vol)
vol_name = 'example_func'
# Step 3: Convert DICOM to NIFTI using dcm2niix
subprocess.run([f'dcm2niix -z n -f {vol_name} -o {example_func_dir} -s y {ref_vol}'], shell = True)
nifti_file = os.path.join(example_func_dir , vol_name + '.nii') # To save each vol as .nii instead to .nii.gz to load faster
# Step 4: Deoblique correction using AFNI's 3dWarp
deoblique_vol = afni.Warp()
deoblique_vol.inputs.in_file = nifti_file
deoblique_vol.inputs.deoblique = True # Deoblique Nifti files
deoblique_vol.inputs.num_threads = 4 # Set number of threads for processing
deoblique_vol.inputs.outputtype = 'NIFTI'
ref_vol_deoblique_file = os.path.join(example_func_dir, vol_name + '_deoblique.nii')
deoblique_vol.inputs.out_file = ref_vol_deoblique_file
deoblique_vol.run()

# Step 5: Brain extraction using AFNI's Automask with parameters to improve session to session registration of functional data
brainextraction = afni.Automask()
brainextraction.inputs.in_file = ref_vol_deoblique_file
brainextraction.inputs.erode = 1 # Erode the mask inwards to avoid skull and tissue fragments. Check this parameter for each subject based on brain extraction performance during training session.
brainextraction.inputs.clfrac = 0.5 # Sets the clip level fraction (0.1 - 0.9). By default 0.5. The larger the restrictive brain extraction is.
brainextraction.inputs.num_threads = 4 # Set number of threads for processing
brainextraction.inputs.outputtype = 'NIFTI'
brain_file = os.path.join(example_func_dir, vol_name + '_deoblique_brain.nii')
brainmask_file = os.path.join(example_func_dir, vol_name + '_deoblique_brainmask.nii')
brainextraction.inputs.brain_file = brain_file # Just brain's data
brainextraction.inputs.out_file = brainmask_file # Brain binarized mask
brainextraction.run()
