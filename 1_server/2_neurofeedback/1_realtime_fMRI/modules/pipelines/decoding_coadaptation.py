############################################################################
# AUTHORS: Najemeddine Abdennour
# EMAIL: nabdennour@bcbl.eu
# COPYRIGHT: Copyright (C) 2024-2025, pyDecNef
# URL: https://github.com/najemabdennour/pyDecNef
# INSTITUTION: Basque Center on Cognition, Brain and Language (BCBL), Spain
# LICENCE: GNU General Public License v3.0
############################################################################
"""
Description:

This script performs decoder coadaptation for neuroimaging data analysis. 
It adaptively updates a machine learning model (decoder) based on incoming neuroimaging 
volumes that meet specific criteria, allowing the model to generalize better across 
different data examples. The process involves loading existing models, evaluating 
new data points, and updating the model accordingly.

The script processes neuroimaging data through several key steps:
    1. Loads existing coadaptation data (if available) or initializes new training
    2. Loads the base model for decoding from the variously supported classification algorithms (e.g., SVM, Random Forests)
    3. Evaluates each incoming volume to determine if it meets the adaptation criteria
    4. Updates the training dataset and labels with qualifying volumes
    5. Re-trains the model on the updated dataset while maintaining separate models for active and background adaptation
    6. Saves the new, adapted model

Inputs:
    - vol_list: List of neuroimaging volumes (NifTI format)
    - ground_truth: Ground truth label for each volume in vol_list
    - model_file: Path to the base decoder model (serialized model)

Outputs:
    - Adapted model: Updated decoder trained on new data points that meet criteria
    - Training data and labels: Extended datasets used for retraining

The script efficiently handles model adaptation by:
    - Only processing volumes that meet the coadaptation criteria
    - Utilizing existing pre-trained models as a foundation
    - Saving updated models with distinct naming schemes
    - Supporting parallel processing where possible
    - Ensuring compatibility with various machine learning algorithms
"""
#############################################################################################
# IMPORT DEPENDENCIES
#############################################################################################

import pandas as pd
import os
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier 
import numpy as np
from modules.config.exp_config import Exp
from joblib import load, dump
from sklearn.calibration import CalibratedClassifierCV
from nilearn.maskers import NiftiMasker

#############################################################################################
# DESCRIPTION
#############################################################################################

# Decoder coadaptation procedure  

#############################################################################################
# Initialize variables
#############################################################################################

classification_method = Exp.classifier_type
coadaptation_data_dir = Exp.coadaptation_training_data_dir
masker = NiftiMasker(Exp.mask_file,).fit()
coadaptation_training_data = masker.transform(Exp.coadaptation_training_data_file)
coadaptation_training_labels = pd.read_csv(Exp.coadaptation_training_data_labels_file)
labels = coadaptation_training_labels["target_category"].values
idx = labels != 2 # discard noise examples
training_data = coadaptation_training_data[idx] # the decoder training data
training_labels = labels[idx] # the decoder training labels

#############################################################################################
# FUNCTION
#############################################################################################

def coadaptation(vol_list,ground_truth,model_file):
    coadaptation_data_path = os.path.join(coadaptation_data_dir,"co_adaptation_data")
    coadaptation_labels_path = os.path.join(coadaptation_data_dir,"co_adaptation_labels")
    if os.path.exists(coadaptation_data_path):
        print("Loading the previous volumes that passed the coadaptation criteria")
        full_training_data = load(coadaptation_data_path) # loading the training data & the vols that passed the coadaptation criteria
        full_training_labels = load(coadaptation_labels_path)# loading the training labels & the labels of the vols that passed the coadaptation criteria
    else:
        full_training_data = training_data # loading the training data
        full_training_labels = training_labels # loading the training labels
    print("the shape of the data loaded for coadaptation:", full_training_data.shape)
    no_coadapted_model = load(model_file) # load the non coadapted decoder
    for vol in vol_list:
        generated_porbs = no_coadapted_model.predict_proba(vol) # predicting the label of the new vols
        decoding_prob = generated_porbs[0][int(ground_truth)] 
        if decoding_prob > Exp.coadaptation_vol_acceptance_criteria: # checking if the vol decoding porb passed the coadaptation criteria
            full_training_data = np.vstack([full_training_data,vol]) # adding vol to training data
            full_training_labels = np.append(full_training_labels,ground_truth) # adding vol label to training labels
            dump(full_training_data,coadaptation_data_path) # saving the new dataset 
            dump(full_training_labels,coadaptation_labels_path) # saving the new labels
    
    # preparing new decoder for training to co-adapt the decoder on the new dataset and labels
    if classification_method == "svm":
        pipeline = make_pipeline(LinearSVC(dual = True,C = int(1),class_weight = 'balanced',random_state = 12345))
    elif classification_method == "svmlinear":
        pipeline = make_pipeline(LinearSVC(penalty = 'l1',dual = False,C = int(1),class_weight = 'balanced',random_state = 12345))
        pipeline = CalibratedClassifierCV(pipeline)
    elif classification_method == "decisiontree":
        pipeline = make_pipeline(DecisionTreeClassifier())
    elif classification_method == "extratree":
        pipeline = make_pipeline(ExtraTreeClassifier())
    elif classification_method == "randomforest":
        pipeline = make_pipeline(RandomForestClassifier())
    elif classification_method == "extratrees":
        pipeline = make_pipeline(ExtraTreesClassifier())
    elif classification_method == "bagging":
        pipeline = make_pipeline(BaggingClassifier())
    elif classification_method == "gradientboosting":
        pipeline = make_pipeline(GradientBoostingClassifier())
        pipeline = make_pipeline(AdaBoostClassifier())
    elif classification_method == "naivebayes":
        pipeline = make_pipeline(GaussianNB())
    elif classification_method == "kneighbors":
        pipeline = make_pipeline(KNeighborsClassifier())
    elif classification_method == "mlp":
        pipeline = make_pipeline(MLPClassifier())
    elif classification_method == "sgd":
        pipeline = make_pipeline(SGDClassifier())
    elif classification_method == "logisticregression":
        pipeline = make_pipeline(LogisticRegression())

    pipeline.fit(full_training_data,full_training_labels) # training the decoder on the new dataset and labels
    if (Exp.coadaptation_active or Exp.coadaptation_background_warmup):
        if "_coadapted" in model_file:
            coadapted_model_path = model_file
        else:
            coadapted_model_path = str(model_file) +"_coadapted"
        dump(pipeline,coadapted_model_path) # saving the new coadapted decoder with a new naming scheme
        if Exp.coadaptation_active:
            print(f"The updated coadaptation model saved in: {coadapted_model_path}")
        else:
            print(f"The updated background warmup coadaptation model saved in: {coadapted_model_path}")
