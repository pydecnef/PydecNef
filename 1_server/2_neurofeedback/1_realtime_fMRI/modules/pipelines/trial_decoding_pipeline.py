############################################################################
# AUTHORS: Pedro Margolles & David Soto
# EMAIL: pmargolles@bcbl.eu, dsoto@bcbl.eu
# COPYRIGHT: Copyright (C) 2021-2022, pyDecNef
# URL: https://pedromargolles.github.io/pyDecNef/
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
Description:

This script performs trial-level fMRI data decoding for neuroimaging analysis, 
specifically designed for real-time applications. It processes preprocessed 
neuroimaging volumes (e.g., fMRI) through multiple decoding approaches to 
classify or predict cognitive states, brain states, or any target 
conditions of interest. The script supports various machine learning 
algorithms and provides flexible processing options based on trial 
characteristics.

Key Features:
    - Supports real-time trial-level fMRI data decoding
    - Implements dynamic decoding for immediate feedback in experimental setups
    - Allows for both single-trial and aggregated (HRF-peak) decoding approaches
    - Compatible with multiple machine learning algorithms (e.g., Logistic Regression, SVM)
    - Includes time-efficient processing to handle rapid data streams
    - Provides detailed timing information for each decoding step


Inputs:
    - preproc_vols_data: List of masked and preprocessed fMRI volumes (NifTI format)
    - model_file: Path to a pretrained machine learning model (e.g., Logistic Regression, SVM)
    - ground_truth: Target class label corresponding to the ground truth condition
   (integer value)

Outputs:
    - decoding_prob: Probability of correct classification for the given volume/condition
    - decoding_time: List containing timing information for each decoding step
    - vols_decoding_probs: Optional output with individual probabilities for each HRF peak volume

Performance Considerations:

The script is optimized for efficient processing, ensuring minimal latency 
between data acquisition and model evaluation. It supports both single-trial 
dynamic decoding and aggregated decoding approaches, allowing flexibility in 
experimental design. The use of pre-trained models and efficient prediction 
algorithms ensures that even high-resolution neuroimaging data can be processed 
quickly.
"""
#############################################################################################
# IMPORT DEPENDENCIES
#############################################################################################

from collections import OrderedDict
from joblib import load
import numpy as np
from sklearn.linear_model import LogisticRegression
#import tensorflow as tf
import time
from typing import Tuple

#############################################################################################
# DESCRIPTION
#############################################################################################

# Decoding pipeline to be applied in real-time on each trial data

#############################################################################################
# FUNCTIONS
#############################################################################################

def average_hrf_peak_vols_decoding(preproc_vols_data: list, model_file: str, ground_truth: int) -> Tuple[np.array, list]:

    """ 
    Average volumes within a trial HRF peak before decoding a single averaged volume to increase signal-to-noise ratio

    Steps:

        1 - Load model
        2 - Average volumes of interest (i.e., volumes within HRF peak) onto a single volume
        3 - Predict class probabilities over this trial averaged volume
        4 - Get probability of decoding ground truth class

    Inputs:

        preproc_vols: masked and preprocessed volumes (List of Numpy arrays. Each array has following dimensions: 1, n_voxels)
        model_file: pretrained model (ex., a classifier) in a Region Of Interest (Have to match with mask_file)
        ground_truth: as integer. Target class position in model probabilities predictions tuple.
        
    Returns:

        decoding_prob: probability of decoding ground truth of averaged HRF volume (Float)
        decoding_time: processing time for each decoding pipeline step (List)
    """

    # Decoding start time
    decoding_time = OrderedDict()
    # Load pretrained model
    clf_fit = load(model_file)
    
    start_decoding = time.time()

    # First, average vols within HRF peak
    start_average_vols = time.time()
    average_preproc = np.average(preproc_vols_data, axis = 0) # Average HRF peak vols
    average_preproc = average_preproc.reshape(1, -1) # Reshape so averaged array has (1, n_voxels) dimensions
    decoding_time['average_vols_time'] = time.time() - start_average_vols
     
    # Predict class probability
    start_prediction = time.time()
    class_probabilities = clf_fit.predict_proba(average_preproc) # Predict class probabilities over this trial averaged vol
    evaluation = clf_fit.evaluate(average_preproc,[ground_truth for i in len(average_preproc)])
    decoding_prob = class_probabilities[0][ground_truth] # Select the probability corresponding to the ground truth
    print("THE DECODING PROBABILITY:",decoding_prob)
    decoding_time['prediction_time'] = time.time() - start_prediction

    # Decoding end time
    decoding_time['total_decoding_time'] = time.time() - start_decoding
    decoding_time = [decoding_time] # Convert dict to a list to store all times in a pandas dataframe

    return decoding_prob, decoding_time



def average_probs_decoding(preproc_vols_data: list, model_file: str, ground_truth: int) -> Tuple[np.array, list, list]:

    """
    Average decoding probabilities of volumes within a trial HRF peak

    Steps:

        1 - Load model
        2 - Predict class probabilities for each HRF peak volume
        3 - Get probability of decoding ground truth class for each HRF peak volume
        4 - Average probabilities of decoding ground truth across volumes

    Inputs:

        preproc_vols: masked and preprocessed volumes (List of Numpy arrays. Each array has following dimensions: 1, n_voxels)
        model_file: pretrained model (ex., a classifier) in a Region Of Interest (Have to match with mask_file)
        ground_truth: as integer. Target class position in model probabilities predictions tuple.
        
    Returns:

        averaged_decoding_prob: averaged decoding probability across HRF volumes (Float)
        vols_decoding_probs: decoding probability for each independent volume within HRF peak (Float)
        decoding_time: processing time for each decoding pipeline step (List)
    """

    # Decoding start time
    decoding_time = OrderedDict()
    # Load pretrained model
    clf_fit = load(model_file)

    start_decoding = time.time()
    start_decoding_vols = time.time()
    vols_decoding_probs = [] # To store each HRF peak vol ground truth decoding probability
        
    # Iterate over all preprocessed volumes within HRF peak
    for vol in preproc_vols_data:
        # Predict class probability
        class_probabilities = clf_fit.predict_proba(vol) # Predict class probabilities over this trial averaged vol
        decoding_prob = class_probabilities[0][int(ground_truth)] # Select the probability corresponding to the ground truth
        vols_decoding_probs.append(decoding_prob) # Append decoding probability to list of decoding probabilities for each vol
    print("THE TOTAL DECODING PROBABILITIES FOR THE VOLUMES:",vols_decoding_probs)   
    averaged_decoding_prob = np.average(vols_decoding_probs) # Average probabilities of decoding ground truth across volumes
    decoding_time['decoding_vols_time'] = time.time() - start_decoding_vols

    # Decoding end time
    decoding_time['total_decoding_time'] = time.time() - start_decoding
    decoding_time = [decoding_time] # Convert dict to a list to store all times in a pandas dataframe

    return averaged_decoding_prob, vols_decoding_probs, decoding_time



def dynamic_decoding(preproc_vol: np.array, model_file: str, ground_truth: int)  -> Tuple[np.array, list]:

    """
    Decode a single volume. In dynamic decoding, all volumes within a trial HRF peak, will be decoded independently and 
    sent individually to experimental software as feedback.

    Steps:

        1 - Load model
        2 - Predict class probabilities of a preprocessed volume
        3 - Get probability of decoding ground truth class for a preprocessed volume

    Inputs:

        preproc_vol: a single masked and preprocessed fMRI volume data (Array with following dimensions: 1, n_voxels)
        model_file: pretrained model (ex., a classifier) in a Region Of Interest (Have to match with mask_file)
        ground_truth: as integer. Target class position in model probabilities predictions tuple.
   
    Returns:

        decoding_prob: probability of decoding ground truth in preproc_vol
        decoding_time: processing time for each decoding pipeline step (List)
    """

    # Decoding start time
    decoding_time = OrderedDict()
    # Load pretrained model
    clf_fit = load(model_file)

    start_decoding = time.time()

    # Predict class probability
    start_prediction = time.time()

    class_probabilities = clf_fit.predict_proba(preproc_vol) # Predict class probabilities over this trial averaged vol
    decoding_prob = class_probabilities[0][ground_truth] # Select the probability corresponding to the ground truth
    print("THE DECODING PROBABILITY:",decoding_prob)   
    decoding_time['prediction_time'] = time.time() - start_prediction

    # Decoding end time
    decoding_time['total_decoding_time'] = time.time() - start_decoding
    decoding_time = [decoding_time] # Convert dict to a list to store all times in a pandas dataframe

    return decoding_prob, decoding_time
