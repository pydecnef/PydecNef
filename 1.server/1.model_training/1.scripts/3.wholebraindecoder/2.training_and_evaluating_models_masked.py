 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from sklearn.model_selection import check_cv
try:
    from nilearn.maskers import NiftiMasker
except:
    from nilearn.input_data import NiftiMasker
from nilearn.image import load_img

import joblib
if __name__ == "__main__":

########################################################
### najem addons
    from sklearn.calibration import CalibratedClassifierCV
    classifiers = ["svm","svmlinear","decisiontree","extratree","randomforest", "extratrees","bagging","gradientboosting","adaboost","naivebayes","kneighbors","mlp","sgd","logisticregression","xgboost","evaluating_existing_model" ]
    # accuracy starts by the best is 14(xgboost),5(extratrees),4(randomforest),13(logisticregression)
    classification_method = classifiers[5]
    preprocessing = True
    f_score_res = []
    import getopt, sys
    def arg_parsing():
        try:
            opts, args = getopt.getopt(sys.argv[1:], "0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15", classifiers)
        except getopt.GetoptError as err:
        # print help information and exit:
            print(str(err)) # will print something like "option -a not recognized"
            sys.exit(2)
        if len(args) < 1:
            return args
        elif args[0] not in [str(i) for i in  range(0,len(classifiers))]:
            assert False, f"unhandled option, only {len(classifiers)} are available, you can pass them by number or name starting with 0 to {len(classifiers)-1} "
        return args
    script_arg =  arg_parsing()
    if script_arg:
        classification_method = classifiers[int(script_arg[0])]
        print("The chosen classification method is: ", classification_method)
    else: 
        print("The chosen classification method is the default: ", classification_method)
########################################################

    if os.path.abspath(__file__).endswith("train.py"):
        exp_dir = Path(os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir) ))
    else:
        exp_dir = Path(os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) ))
    print(exp_dir)
    data_dir = exp_dir / 'preprocessed/stacked_vols_of_interest/'
    working_data = str(data_dir / 'examples_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz')
    wholebrain_mask = os.path.join(str(exp_dir), "preprocessed", "example_func","example_func_deoblique_brainmask.nii")
    #wholebrain_mask = os.path.join(str(exp_dir), "preprocessed", "rois","bilateral_mask_adapted.nii")
    wholebrain_mask = os.path.join(str(exp_dir), "preprocessed", "rois","full_mask.nii.gz")
    #wholebrain_mask = os.path.join(str(exp_dir), "preprocessed", "rois","lingual_native.nii.gz")
    #wholebrain_mask = os.path.join(str(exp_dir), "preprocessed", "rois","occcipitalfusiform_native.nii.gz")
    #wholebrain_mask = os.path.join(str(exp_dir), "preprocessed", "rois","tempoccipFusiform_native.nii.gz")

    masker = NiftiMasker(wholebrain_mask,).fit()
    df = pd.read_csv(data_dir / 'examples_detrended_zscored_stacked_vols_of_interest_searchlight.csv')
    df["groups"] = df["run"] + "_" + df["trial_idx"].values.astype(str) # creating unique trial IDs
    print("trials unique values:\n",df["groups"].unique())
    print("Total nb trials:",len(df["groups"].unique()))
    #########################################################################
    BOLD_signals = masker.transform(working_data) # vectorize the whole brain data
    labels = df["target_category"].values
    groups = df["run"].values #df["groups"].values
    print("The number of runs grouped :", np.unique(groups))
    print("The number of groups for the leave one out cross-validation:", len(np.unique(groups)))
    #groups = df["groups"].values
    idx = labels != 2 # discard noise examples
    wholebrain_data = BOLD_signals[idx]
    wholebrain_labels = labels[idx]
    wholebrain_groups = groups[idx]

    print("the shape of the wholebrain_data:",wholebrain_data.shape)
    #########################################
    ### najem addons
    #from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier 
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
    #############################################



    #########################################################################
    # cross validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_validate,GridSearchCV, StratifiedShuffleSplit, GroupShuffleSplit,LeaveOneGroupOut, LeavePGroupsOut
    from sklearn.metrics import roc_auc_score
    #cv = StratifiedShuffleSplit(n_splits=3, test_size = 0.3, random_state=0)
    #cv = GroupShuffleSplit(n_splits=3,test_size= 0.3,random_state=0 ) 
    cv = LeaveOneGroupOut()
    #cv = LeavePGroupsOut(n_groups=2)
    res = []

########################################################
### najem addons
    if classification_method == "svm":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),StandardScaler(),SVC(#penalty = 'l1',
                                        #dual = False,
                                        #C = int(1),
                                        probability = True,
                                        class_weight = 'balanced',
                                        random_state = 12345,
                                        ))
        else:
            pipeline = make_pipeline(LinearSVC(#penalty = 'l1',
                                        dual = True,C = int(1),
                                        class_weight = 'balanced',
                                        random_state = 12345,
                                        ))
    elif classification_method == "svmlinear":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),StandardScaler(),LinearSVC(#penalty = 'l1',
                                        dual = True,
                                        C = int(1),
                                        class_weight = 'balanced',
                                        random_state = 12345,
                                        ))
            pipeline = CalibratedClassifierCV(pipeline)
        else:
            pipeline = make_pipeline(LinearSVC(penalty = 'l1',dual = False,C = int(1),
                                        class_weight = 'balanced',
                                        random_state = 12345,
                                        ))
            pipeline = CalibratedClassifierCV(pipeline)


    elif classification_method == "decisiontree":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             DecisionTreeClassifier()
                              )
        else:
            pipeline = make_pipeline(DecisionTreeClassifier())

    elif classification_method == "extratree":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             ExtraTreeClassifier()
                              )
        else:
            pipeline = make_pipeline(ExtraTreeClassifier())
    elif classification_method == "randomforest":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             RandomForestClassifier()
                              )
        else:
            pipeline = make_pipeline(RandomForestClassifier())
    elif classification_method == "extratrees":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             ExtraTreesClassifier()
                              )
        else:
            pipeline = make_pipeline(ExtraTreesClassifier())
    elif classification_method == "bagging":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             BaggingClassifier()
                              )
        else:
            pipeline = make_pipeline(BaggingClassifier())
    elif classification_method == "gradientboosting":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             GradientBoostingClassifier()
                              )
        else:
            pipeline = make_pipeline(GradientBoostingClassifier())
    elif classification_method == "adaboost":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             AdaBoostClassifier()
                              )
        else:
            pipeline = make_pipeline(AdaBoostClassifier())
    elif classification_method == "naivebayes":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             GaussianNB()
                              )
        else:
            pipeline = make_pipeline(GaussianNB())
    elif classification_method == "kneighbors":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             KNeighborsClassifier()
                              )
        else:
            pipeline = make_pipeline(KNeighborsClassifier())
    elif classification_method == "mlp":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             MLPClassifier()
                              )
        else:
            pipeline = make_pipeline(MLPClassifier())
    elif classification_method == "sgd":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             SGDClassifier()
                              )
        else:
            pipeline = make_pipeline(SGDClassifier())
    elif classification_method == "logisticregression":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                             LogisticRegression()
                              )
        else:
            pipeline = make_pipeline(LogisticRegression())
    elif classification_method == "xgboost":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             # PCA(random_state = 12345),
                            XGBClassifier()
                          )
        else:
            pipeline = make_pipeline(XGBClassifier())
    elif classification_method == "evaluating_existing_model":
        index = sys.argv[2]
        if int(index) in [1,2,3]:
            model_file_name = f"sklearn_decoder_{index}"
            model_path = os.path.join(str(exp_dir), "models", model_file_name)
            if os.path.exists(f"evaluated_model{index}"):
                os.remove(f"evaluated_model{index}")
            shutil.copy(model_path,f"evaluated_model{index}")
            pipeline =  joblib.load(f"evaluated_model{index}")
        else:
            if Path.exists(Path("evaluated_model")):
                pipeline =  joblib.load("evaluated_model")
            elif Path.exists(Path("evaluated_model1")):
                pipeline =  joblib.load("evaluated_model1")
            elif Path.exists(Path("evaluated_model2")):
                pipeline =  joblib.load("evaluated_model2")
            elif Path.exists(Path("evaluated_model3")):
                pipeline =  joblib.load("evaluated_model3")
            else:
                print("No file for the evaluated_model found")
    else:
        print("please choose a classification_method")
        raise ValueError("No classifier chosen")    
########################################################
    for (train,test), indx in zip(cv.split(wholebrain_data,wholebrain_labels,groups = wholebrain_groups), range(0,len(np.unique(groups))) ):
        if classification_method != "evaluating_existing_model":
            pipeline.fit(wholebrain_data[train],wholebrain_labels[train])
        y_test = wholebrain_labels[test]
        y_pred = pipeline.predict(wholebrain_data[test])
        res.append(roc_auc_score(y_test,y_pred))
        f_score_res.append(f1_score(y_test,y_pred))
        if classification_method != "evaluating_existing_model":
            model_file_name = f"sklearn_decoder_{indx}"
            model_path = os.path.join(str(exp_dir), "models", model_file_name)
            joblib.dump(pipeline,model_path)

    if classification_method != "evaluating_existing_model":  
        pipeline.fit(wholebrain_data,wholebrain_labels)
        model_file_name = f"sklearn_decoder_full_data"
        model_path = os.path.join(str(exp_dir), "models", model_file_name)
        joblib.dump(pipeline,model_path)
########################################################
### najem addons
    if classification_method == "evaluating_existing_model":
        print(f_score_res)
        res = res[int(index)-1]
        f_score_res = f_score_res[int(index)-1]
    else:
        print("accuracy(f1 score):",f_score_res,np.mean(f_score_res))
        print("roc_auc_score:",res,np.mean(res))
    acc = np.mean(f_score_res)
    roc_auc = np.mean(res)
    print(f"\n******** THE ACCURACY: {acc} | THE ROC_AUC: {roc_auc} ********")
########################################################
    if classification_method != "evaluating_existing_model":
        print(f"the last model saved in: {model_path}")
    else:
        print("evaluation complete")

 
    info_df = pd.DataFrame()
    info_df["cv_split"] = [cv.__repr__().replace("()","")]
    info_df["classifier"] =  [classification_method]
    try:
        info_df["pipeline_details"] =  [list(pipeline.named_steps.keys())]
    except:
        info_df["pipeline_details"] =  ["LinearSVC calibrated"]

    if classification_method != "evaluating_existing_model":
        info_df["saved_model_path"] =  [model_path.replace(model_file_name,"")]
        info_df["model_name"] = model_file_name 
        info_df["cv_folds_acc"] = [f_score_res]
        info_df["cv_folds_roc_auc"] = [res]

    else:
        info_df["saved_model_path"] =  [os.getcwd()]
        info_df["model_name"] = [f"evaluated_model{index}"]
    info_df["f1_score_acc"] = acc
    info_df["roc_auc"] = roc_auc
    try:
        df_info = pd.read_csv("info.csv")
        df_info = pd.concat([df_info,info_df]).reset_index(drop=True)
        df_info.to_csv("info.csv",index=False)
    except:
        df_info = info_df
        df_info.to_csv("info.csv",index=False)
if classification_method == "evaluating_existing_model":
    print("THE INFO DATAFRAME:\n",df_info)
