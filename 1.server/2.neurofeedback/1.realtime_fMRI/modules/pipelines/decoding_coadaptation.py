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
cv = StratifiedShuffleSplit(n_splits=10, test_size = 0.3, random_state=0)

#### najem addon
from modules.config.exp_config import Exp
from joblib import load, dump
from sklearn.calibration import CalibratedClassifierCV

nb_vol_passed_coadapt_criteria = 0
classifiers = ["svm","svmlinear","randomforest", "extratrees","logisticregression" ]
    # accuracy starts by the best is 14(xgboost),5(extratrees),4(randomforest),13(logisticregression)
classification_method = classifiers[4]

exp_dir = Path().absolute().parent.parent
model_training_dir = os.path.join(exp_dir,"0.training")
#print(exp_dir)
data_dir =  os.path.join(model_training_dir, 'preprocessed','stacked_vols_of_interest')
#print(data_dir)
    
working_data = os.path.join(data_dir ,'examples_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz')
#wholebrain_mask = os.path.join(model_training_dir, "preprocessed", "example_func","example_func_deoblique_brainmask.nii")
#wholebrain_mask = os.path.join(model_training_dir, "preprocessed", "rois","tempoccipFusiform_native.nii")
wholebrain_mask = Exp.mask_file
#wholebrain_mask = Path("/Users/brain/Downloads/co-adaptation_test/perception/preprocessed/rois/bilateral_mask_adapted.nii")
#masker = NiftiMasker(wholebrain_mask,).fit()
#df = pd.read_csv(os.path.join(data_dir,'examples_detrended_zscored_stacked_vols_of_interest_searchlight.csv'))
df = pd.read_csv(os.path.join(data_dir,'perception_detrended_zscored_stacked_vols_of_interest.csv'))
#print(df.shape)
#BOLD_signals = masker.transform(working_data) # vectorize the whole brain data
#print("the shape of the data:", BOLD_signals.shape)
working_data = os.path.join(data_dir, 'perception_detrended_zscored_stacked_vols_of_interest_bilateral.npy')
BOLD_signals = np.load(working_data) # vectorize the whole brain data
labels = df["target_category"].values
idx = labels != 2 # discard noise examples
wholebrain_data = BOLD_signals[idx]
wholebrain_labels = labels[idx]
#print(wholebrain_data.shape)
#print(wholebrain_labels.shape)
#print(type(wholebrain_labels))
#print(wholebrain_labels)
def coadaptation(vol_list,ground_truth,model_file):
    coadaptation_data_path = os.path.join(data_dir,"co_adaptation_data")
    coadaptation_labels_path = os.path.join(data_dir,"co_adaptation_labels")
    if os.path.exists(coadaptation_data_path):
        print("Loading the previous volumes that passed the coadaptation criteria")
        wholebrain_new_data = load(coadaptation_data_path)
        wholebrain_new_labels = load(coadaptation_labels_path)
    else:
        wholebrain_new_data = wholebrain_data
        wholebrain_new_labels = wholebrain_labels
    print("the shape of the data loaded for coadaptation:", wholebrain_new_data.shape)
    no_coadapted_model = load(model_file) 
    for vol in vol_list:
        generated_porbs = no_coadapted_model.predict_proba(vol)
        decoding_prob = generated_porbs[0][int(ground_truth)]
        if decoding_prob > Exp.coadaptation_criteria:
            wholebrain_new_data = np.vstack([wholebrain_new_data,vol])
            wholebrain_new_labels = np.append(wholebrain_new_labels,ground_truth)
            dump(wholebrain_new_data,coadaptation_data_path)
            dump(wholebrain_new_labels,coadaptation_labels_path)
        

    if classification_method == "svm":
        pipeline = make_pipeline(SVC(probability=True, class_weight = 'balanced',random_state = 12345,)) #penalty = 'l1',dual = False,C = int(1),
    if classification_method == "svmlinear":
        pipeline = make_pipeline(LinearSVC(dual = True,C = int(1), class_weight = 'balanced',random_state = 12345))
        pipeline = CalibratedClassifierCV(pipeline)
    elif classification_method == "randomforest":
        pipeline = make_pipeline(RandomForestClassifier())
    elif classification_method == "extratrees":
        pipeline = make_pipeline(ExtraTreesClassifier())
    elif classification_method == "logisticregression":
        pipeline = make_pipeline(LogisticRegression())

### filtering criteria
    #coadapt_generated_porbs = pipeline.predict_proba(wholebrain_new_data)
    #coadaptation_labels = pd.DataFrame(coadapt_generated_porbs,columns=["non living","living"])
    #filter = [1,None,None]
    #conditions = [coadaptation_labels["living"]> Exp.coadaptation_criteria,coadaptation_labels["non living"]> Exp.coadaptation_criteria,( (coadaptation_labels["living"] <= Exp.coadaptation_criteria) & (coadaptation_labels["non living"] >= (1-Exp.coadaptation_criteria)) )]
    #coadaptation_labels["class"] = np.select(conditions,filter)
    #coadaptation_generated_labels =  coadaptation_labels[~coadaptation_labels["class"].isna()]["class"].reset_index(drop=True)
    #coadaptation_data = wholebrain_new_data[~coadaptation_labels["class"].isna()]
    #coadapt_generated_labels = coadaptation_generated_labels.astype(int) 
    #print("the number of the coadaptation samples (conforming with the criteria) :",len(coadapt_generated_labels))


    pipeline.fit(wholebrain_new_data,wholebrain_new_labels)
    model_path = model_file
    no_coadapted_model_path = str(model_path) +"_no_coadapted"
    dump(no_coadapted_model,Path(no_coadapted_model_path))
    print(f"the no coadapted model saved in: {no_coadapted_model_path}")
    if Exp.coadaptation_active:
        coadapted_model_path = str(model_path) +"_coadapted"
        dump(pipeline,coadapted_model_path)
        print(f"coadapted model saved in: {coadapted_model_path}")
    else:
        coadapted_model_path = str(model_path) +"_coadapted_background"
        dump(pipeline,Path(coadapted_model_path))
        print(f"coadapted model saved in: {coadapted_model_path}")
