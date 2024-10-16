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

classification_method = Exp.classifier_type
coadaptation_data_dir = Exp.coadaptation_training_data_dir
masker = NiftiMasker(Exp.mask_file,).fit()
coadaptation_training_data = masker.transform(Exp.coadaptation_training_data_file)
coadaptation_training_labels = pd.read_csv(Exp.coadaptation_training_data_labels_file)
labels = coadaptation_training_labels["target_category"].values
idx = labels != 2 # discard noise examples
training_data = coadaptation_training_data[idx]
training_labels = labels[idx]

def coadaptation(vol_list,ground_truth,model_file):
    coadaptation_data_path = os.path.join(coadaptation_data_dir,"co_adaptation_data")
    coadaptation_labels_path = os.path.join(coadaptation_data_dir,"co_adaptation_labels")
    if os.path.exists(coadaptation_data_path):
        print("Loading the previous volumes that passed the coadaptation criteria")
        full_training_data = load(coadaptation_data_path)
        full_training_labels = load(coadaptation_labels_path)
    else:
        full_training_data = training_data
        full_training_labels = training_labels
    print("the shape of the data loaded for coadaptation:", full_training_data.shape)
    no_coadapted_model = load(model_file) 
    for vol in vol_list:
        generated_porbs = no_coadapted_model.predict_proba(vol)
        decoding_prob = generated_porbs[0][int(ground_truth)]
        if decoding_prob > Exp.coadaptation_vol_acceptance_criteria:
            full_training_data = np.vstack([full_training_data,vol])
            full_training_labels = np.append(full_training_labels,ground_truth)
            dump(full_training_data,coadaptation_data_path)
            dump(full_training_labels,coadaptation_labels_path)

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

    pipeline.fit(full_training_data,full_training_labels)
    if (Exp.coadaptation_active or Exp.coadaptation_background_warmup):
        if "_coadapted" in model_file:
            coadapted_model_path = model_file
        else:
            coadapted_model_path = str(model_file) +"_coadapted"
        dump(pipeline,coadapted_model_path)
        if Exp.coadaptation_active:
            print(f"The updated coadaptation model saved in: {coadapted_model_path}")
        else:
            print(f"The updated background warmup coadaptation model saved in: {coadapted_model_path}")
