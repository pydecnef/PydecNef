#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conversion of DICOM files to NIFTI format, deoblique correction,
# brain extraction, and spatial registration (correlation) with a reference volume.
"""
This script converts DICOM files to NIFTI format and performs:
 - Deoblique correction
 - Brain extraction using AFNI Automask
 - Spatial registration against a reference volume (AFNI 3dvolreg)
All processing is done in the provided directory structure.
The script uses AFNI's dcm2niix, 3dWarp, Automask, and 3dvolreg commands.
After processing each volume, it organizes files into run-specific directories
and removes unnecessary intermediate files to save space.
"""
from pathlib import Path
from nipype.interfaces import afni as afni
import shutil
import subprocess
import os
# Define the file structure for organization
exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )    #Path().absolute()
raw_vols_dir = Path(os.path.join(exp_dir, '2_data','raw','func'))
preprocessed_dir = os.path.join(exp_dir,'2_data', 'preprocessed')
func_dir = os.path.join(preprocessed_dir, 'func')
os.makedirs(func_dir,exist_ok=True)
example_func_dir = os.path.join(preprocessed_dir,'example_func')
example_func = os.path.join(example_func_dir,'example_func_deoblique_brain.nii')
for folder in raw_vols_dir.iterdir():
    if folder.is_dir():
        # Create a subdirectory for the current run's data
        run_dir = Path(os.path.join(func_dir , folder.stem) )
        os.makedirs(run_dir,exist_ok=True)
        # Convert each DICOM file to NIFTI and handle deoblique correction
        for vol_file in folder.glob('*.dcm'):
            vol_name = vol_file.stem
            # Convert DICOM to NIFTI using dcm2niix
            subprocess.run([f'dcm2niix -z n -f {vol_name} -o {run_dir} -s y {vol_file}'], shell = True)
            nifti_file = os.path.join(run_dir , vol_name + '.nii' ) # To save each vol as .nii instead to .nii.gz to load faster
            # Deoblique converted Nifti file
            deoblique = afni.Warp() # Use AFNI 3dWarp command
            deoblique.inputs.in_file = nifti_file
            deoblique.inputs.deoblique = True # Deoblique Nifti files
            deoblique.inputs.gridset = example_func # Copy train_reference_vol grid so vols dimensions match between sessions
            deoblique.inputs.num_threads = 4 # Set number of threads for processing
            deoblique.inputs.outputtype = 'NIFTI'
            deoblique_file = os.path.join( run_dir , vol_name + '_deoblique.nii')
            deoblique.inputs.out_file = str(deoblique_file)
            # Brain mask creation for better registration
            deoblique.run()
            # Perform brain extraction using AFNI's Automask command to improve session to session registration of functional data
            brainextraction = afni.Automask() 
            brainextraction.inputs.in_file = deoblique_file
            brainextraction.inputs.erode = 1 # Erode the mask inwards to avoid skull and tissue fragments. Check this parameter for each subject based on brain extraction performance during training session.
            brainextraction.inputs.clfrac = 0.5 # Sets the clip level fraction (0.1 - 0.9). By default 0.5. The larger the restrictive brain extraction is.
            brainextraction.inputs.num_threads = 4 # Set number of threads for processing
            brainextraction.inputs.outputtype = 'NIFTI'
            # Output the brain mask and extracted brain NIFTI file
            brain_file = os.path.join(run_dir , vol_name + '_deoblique_brain.nii')
            brainmask_file = os.path.join(run_dir , vol_name + '_deoblique_brainmask.nii')
            brainextraction.inputs.brain_file = brain_file # Just brain's data
            brainextraction.inputs.out_file = brainmask_file # Brain binarized mask
            brainextraction.run()
            # Spatial registration to align with the reference volume
            volreg = afni.Volreg() # Use AFNI 3dvolreg command
            volreg.inputs.in_file = brain_file
            volreg.inputs.basefile = example_func # Take train_reference_vol as base file for registration
            volreg.inputs.args = '-heptic' # Spatial interpolation
            volreg.inputs.num_threads = 4 # Set number of threads for processing
            volreg.inputs.outputtype = 'NIFTI'
            oned_file = os.path.join(run_dir ,vol_name + '_deoblique_brain_corregister.1D')
            oned_matrix_file = os.path.join(run_dir,vol_name + '_deoblique_brain_corregister.aff12.1D')
            md1d_file = os.path.join(run_dir,vol_name + '_deoblique_brain_corregister_md.1D')
            # Output the corregistered NIFTI file and related parameters
            corregister_file = os.path.join(run_dir , vol_name + '_deoblique_brain_corregister.nii')
            # Save transformation parameters
            volreg.inputs.oned_file = oned_file # 1D movement parameters output file -1Dfile
            volreg.inputs.oned_matrix_save = oned_matrix_file # Save the matrix transformation. -1Dmatrix_save
            volreg.inputs.md1d_file = md1d_file # Max displacement output file -maxdisp1D
            volreg.inputs.out_file = corregister_file # Corregistered vol
            # Execute the registration
            volreg.run()
            # Clean up files not needed for further processing
            for file in run_dir.glob('*'): # To save space, remove all files from preprocessed runs folders which does not contain 'corregister.nii' string in their name
                if 'corregister.nii' not in str(file):
                    file.unlink()