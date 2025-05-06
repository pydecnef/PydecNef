 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IMPORT BASIC DEPENDENCIES
"""
Description:
 This script trains and evaluates machine learning models for neuroimaging data classification. 

 It supports various classifiers including SVM, Decision Trees, Random Forests, etc. 

 The script enables cross-validation to assess model performance and provides detailed evaluation metrics.

Key Features:
    - Supports multiple classification algorithms
    - Implements cross-validation for robust performance estimation
    - Includes preprocessing steps (feature selection and scaling)
    - Provides F1-score and ROC-AUC as primary evaluation metrics
    - Saves trained models and results for reproducibility
    - Handles command-line arguments for model selection

Code Overview:
    1. Command Line Argument Handling
        - Accepts a list of valid classifiers
        - Returns the selected method or defaults to a pre-defined one

    2. Data Loading & Preprocessing
        - Loads neuroimaging data and labels
        - Applies whole-brain mask and transforms data
        - Extracts relevant data based on target category

    3. Model Training & Evaluation
        - Sets up pipeline for different classifiers
        - Performs cross-validation to estimate performance
        - Calculates F1-score and ROC-AUC metrics
        - Saves final model and results

    4. Results & Model Management
        - Outputs mean F1-score and ROC-AUC values
        - Creates detailed info CSV with model details and configuration
        - Handles existing models by loading them directly

"""
import os
import numpy as np
import pandas as pd
import joblib
import getopt, sys
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from nilearn.maskers import NiftiMasker

# Set Variables
preprocessing = True
def arg_parsing(selection_list =[]):
    """
    Function to handle command line arguments for selecting classification method.

    Returns:
        A list containing the selected classification method identifier (as a string).
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "0:1:2:3:4:5:6:7:8:9:10:11:12:13:14", selection_list)
    except getopt.GetoptError as err:
        print(f"Usage: {sys.argv[0]} {err.message}")
        sys.exit(2)

    if len(args) < 1:
        return []

# Check if the first argument is a valid index or method name
    try:
        selected = selection_list[int(args[0])]
        print("selected:",selected)
    except ValueError:
        if args[0] in selection_list:
            selected = args[0]
        else:
            raise ValueError(f"Invalid option: {args[0]}, only values from 0 to {len(selection_list)-1} or the exact name of the classification method are allowed.")

    return selected

def main():
    """
    Main function to execute the classification script.

    Description of Functionality:
        - Handles command line arguments for selecting classification method.
        - Loads and preprocesses neuroimaging data.
        - Performs cross-validation to evaluate different classifiers.
        - Saves results and models accordingly.
    """
    # Initialize variables
    classifiers = ["svm", "svmlinear", "decisiontree", "extratree",
                  "randomforest", "extratrees", "bagging", "gradientboosting",
                  "adaboost", "naivebayes", "kneighbors", "mlp", "sgd",
                  "logisticregression", "evaluating_existing_model"]

    # Parse command line arguments
    script_args = arg_parsing(classifiers)

    if not script_args:
        selected_method = classifiers[5]
        print(f"Default classification method chosen: {classifiers[5]}")
    else:
        selected_method = script_args
        print(f"The chosen classification method is: {selected_method}")

    # Define data directories and load data
    exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )   
    data_dir = os.path.join(exp_dir, "2.data", "preprocessed","stacked_vols_of_interest")
    working_data_path = os.path.join(data_dir, "detrended_zscored_stacked_vols_of_interest.nii.gz")
    wholebrain_mask_path = os.path.join(exp_dir,"2.data","preprocessed", "example_func","example_func_deoblique_brainmask.nii")


    # Load data
    df_labels = pd.read_csv(os.path.join(data_dir, "detrended_zscored_stacked_vols_of_interest_labels.csv"))
    df_labels["groups"] = df_labels["run"].astype(str) + "_" + df_labels["trial_idx"].values.astype(str)
    print(f"Total number of trials: {len(df_labels['groups'].unique())}")

    # Create maskers and transform data
    masker = NiftiMasker(wholebrain_mask_path).fit()
    BOLD_signals = masker.transform(working_data_path)

    # Extract relevant data for classification (discard noise)
    idx = df_labels["target_category"] != 2
    wholebrain_data = BOLD_signals[idx]
    wholebrain_labels = df_labels["target_category"].values[idx]
    wholebrain_groups = df_labels["run"].values[idx]

    print("Shape of wholebrain_data:", wholebrain_data.shape)

    # Cross-validation setup
    cv = LeaveOneGroupOut()

    f_score_res = []
    roc_auc_res = []

    if selected_method == "svm":
        if preprocessing:
            pipeline = make_pipeline(
                feature_selection=VarianceThreshold(),
                scaler=StandardScaler(),
                model=SVC(
                    probability=True,
                    class_weight='balanced',
                    random_state=12345
                )
            )
        else:
            pipeline = make_pipeline(LinearSVC(
                dual=True,
                C=1,
                class_weight='balanced',
                random_state=12345
            ))
    elif selected_method == "svmlinear":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                LinearSVC(
                    penalty='l1',
                    dual=False,
                    C=1,
                    class_weight='balanced',
                    random_state=12345
                )
            )
            pipeline = CalibratedClassifierCV(pipeline)
        else:
            pipeline = make_pipeline(
                LinearSVC(
                    dual=True,
                    C=1,
                    class_weight='balanced',
                    random_state=12345
                )
            )
            pipeline = CalibratedClassifierCV(pipeline)
    elif selected_method == "decisiontree":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                DecisionTreeClassifier()
            )
        else:
            pipeline = DecisionTreeClassifier()
    elif selected_method == "extratree":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                ExtraTreeClassifier()
            )
        else:
            pipeline = ExtraTreeClassifier()
    elif selected_method == "randomforest":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                RandomForestClassifier()
            )
        else:
            pipeline = RandomForestClassifier()
    elif selected_method == "extratrees":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                ExtraTreesClassifier()
            )
        else:
            pipeline = ExtraTreesClassifier()
    elif selected_method == "bagging":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                BaggingClassifier()
            )
        else:
            pipeline = BaggingClassifier()
    elif selected_method == "gradientboosting":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                GradientBoostingClassifier()
            )
        else:
            pipeline = GradientBoostingClassifier()
    elif selected_method == "adaboost":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                AdaBoostClassifier()
            )
        else:
            pipeline = AdaBoostClassifier()
    elif selected_method == "naivebayes":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                GaussianNB()
            )
        else:
            pipeline = GaussianNB()
    elif selected_method == "kneighbors":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                KNeighborsClassifier()
            )
        else:
            pipeline = KNeighborsClassifier()
    elif selected_method == "mlp":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),  
                MLPClassifier()
            )
        else:
            pipeline = MLPClassifier()
    elif selected_method == "sgd":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                SGDClassifier()
            )
        else:
            pipeline = SGDClassifier()
    elif selected_method == "logisticregression":
        if preprocessing:
            pipeline = make_pipeline(
                VarianceThreshold(),
                StandardScaler(),
                LogisticRegression()
            )

    elif selected_method == "evaluating_existing_model":
        # If no existing model, raise error
        evaluated_model_path = os.path.join(exp_dir,"3.models", "wholebrain", "evaluated_model")
        if not os.path.exists(evaluated_model_path):
            print(f"Path: {evaluated_model_path} does not exist. Please ensure the model is saved there with that naming scheme.")
            sys.exit(1)
        else:
            # Load existing model
            pipeline = joblib.load(evaluated_model_path)
    else:
        pipeline = make_pipeline(LogisticRegression())

    # Cross-validation loop
    for train_idx, test_idx in cv.split(wholebrain_data, wholebrain_labels, groups=wholebrain_groups):
        if selected_method != "evaluating_existing_model":
            pipeline.fit(wholebrain_data[train_idx], wholebrain_labels[train_idx])

        y_pred = pipeline.predict(wholebrain_data[test_idx])
        roc_auc = roc_auc_score(wholebrain_labels[test_idx], y_pred)
        f1 = f1_score(wholebrain_labels[test_idx], y_pred, average='macro')
        roc_auc_res.append(roc_auc)
        f_score_res.append(f1)


    # Final evaluation (for non-existing models)
    if selected_method != "evaluating_existing_model":
        pipeline.fit(wholebrain_data, wholebrain_labels)
        os.makedirs(os.path.join(exp_dir, "3.models","wholebrain") ,exist_ok=True)
        model_file_name = f"sklearn_decoder"
        model_path = os.path.join(exp_dir,"3.models", "wholebrain", model_file_name)
        joblib.dump(pipeline, model_path)

        print(f"Final Accuracy (F1-score): {f_score_res,np.mean(f_score_res)}")
        print(f"Final ROC AUC: {roc_auc_res,np.mean(roc_auc_res)}")
        print(f"The last trained model saved at path:\n{model_path}")
    # Save info to CSV
    info_df = pd.DataFrame()
    info_df["cv_split"] = [cv.__repr__().replace("()","")]
    info_df["classifier"] = [selected_method]
    try:
        pipeline_components = pipeline.named_steps.keys()
        info_df["pipeline_components"] = [list(pipeline_components)]
    except AttributeError:
        info_df["pipeline_components"] = ["LinearSVC calibrated"]

    if selected_method != "evaluating_existing_model":
        model_info = os.path.join(exp_dir,"3.models", "wholebrain", "info.csv")
        try:
            df_info = pd.read_csv(model_info)
            df_info = pd.concat([df_info, info_df], ignore_index=True).reset_index(drop=True)
            df_info.to_csv(model_info, index=False)
        except FileNotFoundError:
            info_df.to_csv(model_info, index=False)
        print(f"Training info saved at:\n{model_info}")

    # Print final info
    print(f"\nFinal Results:")
    print(f"Accuracy (F1): {np.mean(f_score_res)}")
    print(f"ROC AUC: {np.mean(roc_auc_res)}")

if __name__ == "__main__":
    main()
