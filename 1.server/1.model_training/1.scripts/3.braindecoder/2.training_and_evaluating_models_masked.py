 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import pandas as pd
import getopt, sys
from sklearn.calibration import CalibratedClassifierCV
from nilearn.maskers import NiftiMasker
import joblib
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
# Set Variables
preprocessing = True
masks = ['full_mask','lingual_native','occcipitalfusiform_native','tempoccipFusiform_native'] # list of mask name used in the ROI masking process
if __name__ == "__main__":
    classifiers = ["svm","svmlinear","decisiontree","extratree","randomforest", "extratrees","bagging","gradientboosting","adaboost","naivebayes","kneighbors","mlp","sgd","logisticregression","evaluating_existing_model" ]
    classification_method = classifiers[5]
    f_score_res = []
    def arg_parsing():
        try:
            opts, args = getopt.getopt(sys.argv[1:], "0:1:2:3:4:5:6:7:8:9:10:11:12:13:14", classifiers)
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

    exp_dir = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,os.pardir) )   
    data_dir = os.path.join(exp_dir,'2.data','preprocessed','stacked_vols_of_interest')
    working_data = os.path.join(data_dir,'detrended_zscored_stacked_vols_of_interest.nii.gz')

    if classification_method == "svm":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),StandardScaler(),SVC(
                                        probability = True,
                                        class_weight = 'balanced',
                                        random_state = 12345,
                                        ))
        else:
            pipeline = make_pipeline(LinearSVC(
                                        dual = True,C = int(1),
                                        class_weight = 'balanced',
                                        random_state = 12345,
                                        ))
    elif classification_method == "svmlinear":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),StandardScaler(),LinearSVC(
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
                             DecisionTreeClassifier()
                              )
        else:
            pipeline = make_pipeline(DecisionTreeClassifier())

    elif classification_method == "extratree":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             ExtraTreeClassifier()
                              )
        else:
            pipeline = make_pipeline(ExtraTreeClassifier())
    elif classification_method == "randomforest":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             RandomForestClassifier()
                              )
        else:
            pipeline = make_pipeline(RandomForestClassifier())
    elif classification_method == "extratrees":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             ExtraTreesClassifier()
                              )
        else:
            pipeline = make_pipeline(ExtraTreesClassifier())
    elif classification_method == "bagging":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             BaggingClassifier()
                              )
        else:
            pipeline = make_pipeline(BaggingClassifier())
    elif classification_method == "gradientboosting":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             GradientBoostingClassifier()
                              )
        else:
            pipeline = make_pipeline(GradientBoostingClassifier())
    elif classification_method == "adaboost":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             AdaBoostClassifier()
                              )
        else:
            pipeline = make_pipeline(AdaBoostClassifier())
    elif classification_method == "naivebayes":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             GaussianNB()
                              )
        else:
            pipeline = make_pipeline(GaussianNB())
    elif classification_method == "kneighbors":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             KNeighborsClassifier()
                              )
        else:
            pipeline = make_pipeline(KNeighborsClassifier())
    elif classification_method == "mlp":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             MLPClassifier()
                              )
        else:
            pipeline = make_pipeline(MLPClassifier())
    elif classification_method == "sgd":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             SGDClassifier()
                              )
        else:
            pipeline = make_pipeline(SGDClassifier())
    elif classification_method == "logisticregression":
        if preprocessing:
            pipeline = make_pipeline(VarianceThreshold(),
                             StandardScaler(),
                             LogisticRegression()
                              )
        else:
            pipeline = make_pipeline(LogisticRegression())
    elif classification_method == "evaluating_existing_model":
        if os.path.exists(os.path.join(exp_dir, "3.models","masked","evaluated_model")):
                pipeline =  joblib.load(os.path.join(exp_dir, "3.models","masked","evaluated_model"))
        else:
            print("No file for the evaluated_model found. Please name the model that you want to evaluate to: 'evaluated_model' and try again.")
    else:
        print("please choose a classification_method")
        raise ValueError("No classifier chosen")    

    for mask in masks:
        try:
            wholebrain_mask = os.path.join(exp_dir,"2.data", "rois_masks",f"{mask}.nii")
            masker = NiftiMasker(wholebrain_mask,).fit()
        except ValueError:
            wholebrain_mask = os.path.join(exp_dir,"2.data", "rois_masks",f"{mask}.nii.gz")
            masker = NiftiMasker(wholebrain_mask).fit()
        df = pd.read_csv(os.path.join(data_dir,'detrended_zscored_stacked_vols_of_interest_labels.csv'))
        df["groups"] = df["run"] + "_" + df["trial_idx"].values.astype(str) # creating unique trial IDs
        BOLD_signals = masker.transform(working_data) # vectorize the whole brain data
        labels = df["target_category"].values
        groups = df["run"].values 
        idx = labels != 2 # discard noise examples
        wholebrain_data = BOLD_signals[idx]
        wholebrain_labels = labels[idx]
        wholebrain_groups = groups[idx]

        cv = LeaveOneGroupOut()
        res = []
        if classification_method == "evaluating_existing_model":

            try:
                for (train,test), indx in zip(cv.split(wholebrain_data,wholebrain_labels,groups = wholebrain_groups), range(0,len(np.unique(groups))) ):
                    y_test = wholebrain_labels[test]
                    y_pred = pipeline.predict(wholebrain_data[test])

                    res.append(roc_auc_score(y_test,y_pred))
                    f_score_res.append(f1_score(y_test,y_pred))
                print("Total nb trials:",len(df["groups"].unique()))
                print("The number of runs grouped :", np.unique(groups))
                print("The number of groups for the leave one out cross-validation:", len(np.unique(groups)))
                print("the shape of the masked_data:",wholebrain_data.shape)
            except ValueError:
                continue
        elif classification_method != "evaluating_existing_model":
            print("Total nb trials:",len(df["groups"].unique()))
            print("The number of runs grouped :", np.unique(groups))
            print("The number of groups for the leave one out cross-validation:", len(np.unique(groups)))
            print("the shape of the masked_data:",wholebrain_data.shape)
            for (train,test), indx in zip(cv.split(wholebrain_data,wholebrain_labels,groups = wholebrain_groups), range(0,len(np.unique(groups))) ):
                pipeline.fit(wholebrain_data[train],wholebrain_labels[train])
                y_test = wholebrain_labels[test]
                y_pred = pipeline.predict(wholebrain_data[test])
                res.append(roc_auc_score(y_test,y_pred))
                f_score_res.append(f1_score(y_test,y_pred))
                model_file_name = f"decoder_{mask}_cv_{indx}"
                os.makedirs(os.path.join(exp_dir, "3.models","masked",mask) ,exist_ok=True)
                model_path = os.path.join(exp_dir, "3.models","masked",mask, model_file_name)
                joblib.dump(pipeline,model_path)
            pipeline.fit(wholebrain_data,wholebrain_labels)
            model_file_name = f"sklearn_decoder_{mask}"
            model_path = os.path.join(exp_dir, "3.models","masked",mask, model_file_name)
            joblib.dump(pipeline,model_path)
            print("accuracy(f1 score):",f_score_res,np.mean(f_score_res))
            print("roc_auc_score:",res,np.mean(res))
            print(f"the last model saved in: {model_path}")
        acc = np.mean(f_score_res)
        roc_auc = np.mean(res)
        print(f"\n******** THE ACCURACY: {acc} | THE ROC_AUC: {roc_auc} ********")
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
            info_df["model_name"] = ["evaluated_model"]
        info_df["f1_score_acc"] = acc
        info_df["roc_auc"] = roc_auc

        try:
            df_info = pd.read_csv(os.path.join(exp_dir, "3.models","masked","info.csv"))
            df_info = pd.concat([df_info,info_df]).reset_index(drop=True)
            df_info.to_csv(os.path.join(exp_dir, "3.models","masked","info.csv"),index=False)
        except:
            df_info = info_df
            df_info.to_csv(os.path.join(exp_dir, "3.models","masked","info.csv"),index=False)
        if classification_method == "evaluating_existing_model":
            print("evaluation complete")
            print("THE INFO DATAFRAME:\n",df_info)
