try:
    from nilearn.maskers import NiftiMasker
except:
    from nilearn.input_data import NiftiMasker
import pandas as pd
import os
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits=3, test_size = 0.3, random_state=0)

classifiers = ["svm","randomforest", "extratrees","logisticregression" ]
    # accuracy starts by the best is 14(xgboost),5(extratrees),4(randomforest),13(logisticregression)
classification_method = classifiers[2]

exp_dir = Path().absolute().parent.parent
model_training_dir = os.path.join(exp_dir,"0.decoder_training")
#print(exp_dir)
data_dir =  os.path.join(model_training_dir, 'preprocessed','stacked_vols_of_interest')
#print(data_dir)
    
working_data = os.path.join(data_dir ,'examples_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz')
wholebrain_mask = os.path.join(model_training_dir, "preprocessed", "example_func","example_func_deoblique_brainmask.nii")
masker = NiftiMasker(wholebrain_mask,).fit()
df = pd.read_csv(os.path.join(data_dir,'examples_detrended_zscored_stacked_vols_of_interest_searchlight.csv'))
#print(df.shape)
BOLD_signals = masker.transform(working_data) # vectorize the whole brain data
labels = df["target_category"].values
idx = labels != 2 # discard noise examples
wholebrain_data = BOLD_signals[idx]
wholebrain_labels = labels[idx]

df["groups"] = df["run"] + df["trial_idx"].values.astype(str) # creating unique trial IDs
groups = df["groups"].values
wholebrain_groups = groups[idx]
#print(wholebrain_data.shape)
#print(wholebrain_labels.shape)
#print(type(wholebrain_labels))
#print(wholebrain_labels)
def coadaptation(vol_list,ground_truth,model_file):
    for (train,test), index in zip(cv.split(wholebrain_data,wholebrain_labels,groups = wholebrain_groups), range(1,4) ):
        if index ==1:
            wholebrain_new_data = wholebrain_data[train]
            wholebrain_new_labels = wholebrain_labels[train]
    #wholebrain_new_data = wholebrain_data
    #wholebrain_new_labels = wholebrain_labels
    for vol in vol_list:
        wholebrain_new_data = np.vstack([wholebrain_new_data,vol])
        wholebrain_new_labels = np.append(wholebrain_new_labels,ground_truth)
    if classification_method == "svm":
        pipeline = make_pipeline(SVC(probability=True, class_weight = 'balanced',random_state = 12345,)) #penalty = 'l1',dual = False,C = int(1),
    elif classification_method == "randomforest":
        pipeline = make_pipeline(RandomForestClassifier())
    elif classification_method == "extratrees":
        pipeline = make_pipeline(ExtraTreesClassifier())
    elif classification_method == "logisticregression":
        pipeline = make_pipeline(LogisticRegression())
    pipeline.fit(wholebrain_new_data,wholebrain_new_labels)
    model_path = model_file
    import joblib
    joblib.dump(pipeline,model_path)
    print(f"model saved in: {model_path}")