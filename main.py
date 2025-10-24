''' 
Main script for KCNQ1 RF classifiers. 
This script calls plot_scripts.py and variant_dataset.py
To execute: python main.py inputs.yaml 
''' 

import yaml
import numpy as np
import sys
import joblib
import h5py

import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold , RepeatedStratifiedKFold, RepeatedKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from collections import defaultdict,Counter

np.set_printoptions(precision=6, suppress=True)

######################################################################
from variant_dataset import VariantDataset, TestDataset
from plot_scripts import *
######################################################################

def main():
    # Read yaml configularion file:
    if len(sys.argv)==2:
        configFile = sys.argv[1]        
        print(f"Executing {configFile}")
    else:
        print("ERROR: No configuration file specified.")
        return
    
    with open(configFile,"r") as f:
        config = yaml.safe_load(f)

    ## Define variables
    featuresCSVList = config["featuresCSVList"]
    trainLabelsCSV = config["trainLabelsCSV"]
    testLabelsCSV = config.get("testLabelsCSV",None)
    # Options to modify the data: 
    excludeWT = config.get("excludeWT",False)
    exclNM = config.get("exclNM", False)
    exclUNC = config.get("exclUNC", False)        
    VSDonly = config.get("VSDonly",False) 
    PDonly = config.get("PDonly",False)

    check_list = config.get("check_list",None)
    model_features = config.get("model_features",None)
    
    ####################    
    ## Procedures
    am_val_plot = config.get("am_val_plot",False) # measurement v pvalue, color by AM prediction
    train_rfc = config.get("train_rfc",False)
    analyze_validation_oneclass = config.get("analyze_validation_oneclass",False)
    analyze_crossval = config.get("analyze_crossval",False)
    gather_val_list_class = config.get("gather_val_list_class",False)
    
    check_exp_data = config.get("check_experiment_data",False)
    enable_train_save = config.get("enable_train_save",False)
    enable_test = config.get("enable_test",False)
    enable_test_noGTlabels = config.get("enable_test_noGTlabels",False)
    
    fname=config.get("fname_prefix",None) #"out_train_val."
    fname_model=config.get("fname_model_prefix",None) #"save_model_state. .."

    ## Read data, perform what I need to. 
    for label in config["labels"]:
        labelName = label["name"]
        rcut = label["rcut"]
        pvalcut = label["pvalcut"]
        gof = label["gof"]
        ds = VariantDataset(featuresCSVList,trainLabelsCSV,labelName,rcut,pvalcut,gof,excludeWT,exclNM,exclUNC,VSDonly,PDonly,None,None)
        #ds_nonpert = VariantDataset(featuresCSVList,trainLabelsCSV,labelName,rcut,pvalcut,gof,"True",VSDonly,PDonly,None,None)

        # Plot measurements, get distributions of variants 
        if check_exp_data:
            checkRatios(ds)
            plotExp(ds,rcut,pvalcut)
            plt.title(labelName)
            plt.show()

        if am_val_plot:
            include_list = config.get("include_list",None)
            if include_list is None:
                dtest = ds
                print("including all variants in feature/train csvs.")
            else: 
                includeVars = np.loadtxt(include_list, dtype=str)
                dtest = VariantDataset(featuresCSVList,trainLabelsCSV,labelName,rcut,pvalcut,gof,excludeWT,exclNM,exclUNC,VSDonly,PDonly,None,includeVars)
            amValPlot(dtest,rcut,pvalcut)
            plt.show()
        
        # RFC hyperparameter tune, then perform cross-validation
        if train_rfc:
            try: # initialize dictionary if not done so
                best_models_hyperparams
            except NameError:
                best_models_hyperparams=defaultdict(list)                
            ofname = fname+labelName+".joblib" if fname is not None else None
            best_models_hyperparams[labelName]=tryRFC(ds,ofname)

        if gather_val_list_class:
            try: # create dictionary if not done already
                allValClassPreds
            except NameError:
                allValClassPreds={}
            allValClassPreds.update(getValListClass(ds,fname,labelName,check_list))

        ##########
        # Train and save model. do not need to evaluate performance of splits. 
        if enable_train_save:
            exclude_list = config.get("exclude_list",None)
            ofname = fname_model+labelName+".joblib" if fname_model is not None else None            
            if exclude_list is not None:
                excludeVars = np.loadtxt(exclude_list, dtype=str)
                dtrain = VariantDataset(featuresCSVList,trainLabelsCSV,labelName,rcut,pvalcut,gof,excludeWT,exclNM,exclUNC,VSDonly,PDonly,excludeVars,None)
            else:
                dtrain = ds
            savedParams = fname+"_hyperparams.joblib" if fname is not None else "" # this fname likely starts with out_
            trainSingle(savedParams,labelName,dtrain,model_features,model_name=ofname)
            
        if enable_test: 
            include_list = config.get("include_list",None)
            if include_list is None:
                print("Warning: need to specify include_list variants for processing.")
            else: 
                includeVars = np.loadtxt(include_list, dtype=str)
                dtest = VariantDataset(featuresCSVList,trainLabelsCSV,labelName,rcut,pvalcut,gof,excludeWT,exclNM,exclUNC,VSDonly,PDonly,None,includeVars)
            print(f"Total # variants in test set: {len(dtest)}")
            savedModel = fname_model+labelName+".joblib" if fname_model is not None else ""
            try: # create alltestresults if not done already
                allTestResults
            except NameError:
                allTestResults={}
            allTestResults.update(testSingle(dtest,savedModel,labelName,model_features))
            
        if enable_test_noGTlabels:
            include_list = config.get("include_list",None)
            includeVars = np.loadtxt(include_list, dtype=str)
            dtest = TestDataset(featuresCSVList,labelName,excludeWT,VSDonly,PDonly,None,includeVars)
            print(f"Total # variants in test set: {len(dtest)}")
            savedModel = fname_model+labelName+".joblib" if fname_model is not None else ""
            try: # create alltestresults if not done already
                allTestResults
            except NameError:
                allTestResults={}
            allTestResults.update(testSingleNoGTLabels(dtest,savedModel,labelName,model_features))
            
        ##########            

    if train_rfc: # if best_models_hyperparams exists, then save to file
        ofname = fname+"_hyperparams.joblib" if fname is not None else None                    
        joblib.dump(best_models_hyperparams, ofname)
        
    if analyze_validation_oneclass:
        models=['am','orig','origam','origammpnnesm']
        #models=['am','ours'] 
        labels = [cfg["name"] for cfg in config["labels"]]
        plotPerformanceMetrics_oneclass(fname,labels,models)
        plt.show()
        
    if analyze_crossval:
        models=['am','orig','origam','origammpnnesm']
        cf_crossvalPreds(fname,models,check_list,featuresCSVList,trainLabelsCSV,config)
            
    # Compare validation variant classification across labels
    if gather_val_list_class:
        labels = [cfg["name"] for cfg in config["labels"]]
        #cmap = plt.get_cmap("Pastel1_r",2)
        cmap = ListedColormap(['whitesmoke', 'powderblue'])
        ax,missing_cells=plotVarClass(labels,allValClassPreds,cmap) # one model only
        for cell in missing_cells:
            i,j = cell["i"], cell["j"]
            variant,labelName = cell["variant"],cell["label"]
            # Get correct rcut, pvalcut, for this labelName
            label = next(label for label in config["labels"] if label["name"] == labelName)
            rcut,pvalcut,gof = label["rcut"], label["pvalcut"], label["gof"]
            dsvar = VariantDataset(featuresCSVList,trainLabelsCSV,labelName,rcut,pvalcut,gof,excludeWT,exclNM,exclUNC,VSDonly,PDonly,None,variant)            
            savedModel = fname_model+labelName+".joblib" if fname_model is not None else ""
            add_missing_heatmap_cells(dsvar,savedModel,ax,cell,cmap)
            plt.savefig('rfc_pred_heatmap.pdf', bbox_inches="tight") 
        plt.show()

    # Plot enable_test results
    if enable_test:
        labels = [cfg["name"] for cfg in config["labels"]]
        cmap = ListedColormap(['whitesmoke', 'powderblue'])
        ax,missing_cells=plotVarClass(labels,allTestResults,cmap)
        ## missing cells: plot measurements and predictions
        plot_missing_variants(missing_cells, config, featuresCSVList, trainLabelsCSV,fname_model,cmap) 
        plt.savefig('output_enable_test.pdf',format='pdf',bbox_inches="tight")
        plt.show()

    if enable_test_noGTlabels:
        labels = [cfg["name"] for cfg in config["labels"]]
        cmap = ListedColormap(['whitesmoke', 'powderblue'])
        plotTestPreds(labels,allTestResults,cmap)
        #plotCategories(labels,allTestResults)
        #plt.show()

        write_tofile = config.get("write_tofile",None)        
        if write_tofile is not None:
            writePredstoHDF5(labels,allTestResults,write_tofile)
            
    return 

##############################
def checkRatios(ds):
    '''
    This function checks ratios in the dataset.
    Prints counts to compare variant location on protein, classifications, and nans.
    '''

    print("total variants=",len(ds))
    print("normal=",sum([(labels==0).item() for _,labels in ds]))
    print("dysfunctional, lof=",sum([(labels==1).item() for _,labels in ds]))    
    print("dysfunctional, gof=",sum([(labels==5).item() for _,labels in ds]))    
    print("likely dysfunctional=",sum([(labels==2).item() for _,labels in ds]))
    print("uncertain=",sum([(labels==4).item() for _,labels in ds]))
    print("nans=",sum([(labels==3).item() for _,labels in ds]))
    print("not measured=",sum([(labels==6).item() for _,labels in ds]))    

    print(" ")
    
    return

##############################
def tryRFC(ds,ofname_data=None,hyperparam_tune=True):
    ''' 
    RFC hyperparam tune. Then train/validation with cross-validation.
    '''

    featuresList=['change no. H acceptor sites','change no. H donor sites','change volume of aa','change PSSM NR','mutat AA hydrophobicity','mutant AA polarizability','functional density (polarizability 6.5 A)','functional density (polarizability 12 A)','functional density (hydrophobicity 1 A)','functional density (hydrophobicity 6.5 A)','distance from channel pore axis','burial on membrane','AM','MPNN','ESM']

    idx0,idx1 = 0,15 # all features 
    # test various feature sets . everything gets passed in (features_np), but only pass ones I want during training 
    featsets=[[0,12],[0,13],[0,15]]
    models=['orig','origam','origammpnnesm']
 
    sorted_residues = sorted(ds.featureDict.keys())
    features_np = np.stack([ds.featureDict[name].detach().cpu().numpy() for name in sorted_residues])[:,idx0:idx1]
    labels_np = (np.stack([ds.labelDict[name].detach().cpu().numpy() for name in sorted_residues]) >= 1).astype(int)

    ####################    
    # grid search with cross-val
    best_models = {}
    for comboModel,featidx in zip(models,featsets):
        features_np_icombo = features_np[:,featidx[0]:featidx[1]]
        if hyperparam_tune:
            best_model = gridSearchRFC(features_np_icombo,labels_np)
        else:
            print("No hyperparameter tuning. Need to explicitly set hyperparameters.")
            
        best_models[comboModel] = best_model
    ####################
        
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=14926) #31133)
    
    # mcc_scores, am_mccs = [],[]
    results = {}    
    for fold, (train_idx, test_idx) in enumerate(kf.split(features_np,labels_np)):
        X_train_orig , y_train_orig = features_np[train_idx], labels_np[train_idx]
        X_test_orig,y_test_orig = features_np[test_idx],labels_np[test_idx]
        
        X_train = X_train_orig
        y_train = y_train_orig

        # Fit on training features, test on test split
        perturb_mask = (X_test_orig[:, 0:4].sum(axis=1) != 0)    # only get perturbing variants for test
        X_test = X_test_orig[perturb_mask]
        y_test = y_test_orig[perturb_mask]
        valid_indices = [test_idx[i] for i in np.where(perturb_mask)[0]]
        valid_vars = [sorted_residues[i] for i in valid_indices]
        results[fold]={}

        ##########
        ###### Fit and pred for each feature combination 
        for comboModel,featidx in zip(models,featsets): # loop over feature combinations 
            pipeline = best_models[comboModel] 
            pipeline.fit(X_train[:,featidx[0]:featidx[1]],y_train)
            y_pred_proba = pipeline.predict_proba(X_test[:,featidx[0]:featidx[1]]) 
            y_pred_class = np.argmax(y_pred_proba, axis=1)  # convert probabilities to class predictions
            mcc_thresh = performance_check(best_models[comboModel],y_test,y_pred_proba,featuresList)            
            results[fold][comboModel] = {
                "validation": {
                    "preds": y_pred_proba,
                    "trues": y_test,
                    "variants": valid_vars
                },
                "mcc_thresh": mcc_thresh
            }
        ## END         for comboModel,featset in zip(models,featsets):

        ##########
        ###### Compare and save pseudo-pred for AM 
        results[fold]["am"] = {
            "validation": {
                "preds": X_test[:,12],
                "trues": y_test
            }
        }
        y_am_value = X_test[:,12]
        y_bin = y_test # (y_test >= 1).astype(int)
        print("AM mcc:",skm.matthews_corrcoef(y_bin,(y_am_value>=0.5).astype(int)), " Brier score:",skm.brier_score_loss(y_test,y_am_value))
        ##########
         
    #print(f"Mean MCC over 5 folds: {np.mean(mcc_scores):.3f}, for AM: {np.mean(am_mccs):.3f}\n")

    ## save cross validation results to file
    if ofname_data is not None:
        joblib.dump(results, ofname_data)
    
    return best_models

##############################
def gridSearchRFC(features, labels):
    # Pipeline: scaling + RF (scaling isn't strictly needed for RF but kept for consistency)
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
    
    # Define hyperparameter grid
    param_grid = {
        'randomforestclassifier__max_depth': [2,5,7], # do not want it super deep
        'randomforestclassifier__n_estimators': [64,128,256,512,1024], #,2048],
        'randomforestclassifier__max_features': [None,'sqrt','log2'],
        'randomforestclassifier__min_samples_leaf': [1, 2,4],    
        'randomforestclassifier__min_samples_split': [2,5],
        'randomforestclassifier__criterion': ['gini','entropy']
    }
    
    ## Scorer
    custom_scorer = skm.make_scorer(custom_metric,needs_proba=True)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=14926) #31133) # echo $RANDOM on inron 
    
    # Grid search
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        #scoring=mcc_scorer,
        scoring=custom_scorer,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(features, labels)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid.best_estimator_

##############################
def custom_metric(y_true,y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    mcc = skm.matthews_corrcoef(y_true, y_pred)
    brier = -skm.brier_score_loss(y_true, y_pred_proba)  # negative for higher=better
    return 0.7*mcc + 0.3*brier

##############################
def performance_check(best_model,trues,preds,featuresList):
    # Just some quick performance checks
    
    ## Performance
    mcc = skm.matthews_corrcoef(trues, np.argmax(preds, axis=1))
    brier = skm.brier_score_loss(trues,preds[:,1])
    print(f"Our model, each class MCC: {mcc:.3f}, Brier score: {brier:.3f}")
    thresholds = np.arange(0.1,0.9,0.01)
    mccThresh =  [skm.matthews_corrcoef(trues, (preds[:,1] > thresh).astype(int)) for thresh in thresholds]
    maxMCC = max(mccThresh)
    maxMCCThresh = thresholds[np.argmax(mccThresh)]
    print(f"  max MCC= {maxMCC}, thresh={maxMCCThresh}")
    
    clf = best_model.named_steps['randomforestclassifier']
    importances = clf.feature_importances_  # shape: (n_features,)
    top6_idx = np.argsort(importances)[-6:][::-1]  # descending order
    for rank, idx in enumerate(top6_idx, 1):
        print(f"  #{rank} {featuresList[idx]}: {importances[idx]:.6f}")
        print("\n")

    return maxMCCThresh
        

##############################
def plotPerformanceMetrics_oneclass(fname_prefix,all_labels,models):
    '''
    Input is joblib file containing dictionary of validation preds and trues per fold. 
    Equivalent to plotPerformanceMetrics_trainWAM(..) for ANN models. 
    '''

    mcc_valid_results = defaultdict(lambda: defaultdict(list))  # Auto-creates nested lists
    brier_valid_results = defaultdict(lambda: defaultdict(list))  # Auto-creates nested lists    
    figroc,axroc = plt.subplots(len(models),len(all_labels),figsize=(16,9)) # ,sharex=True,sharey=True)
    figprc,axprc = plt.subplots(len(models),len(all_labels),figsize=(16,9)) # ,sharex=True,sharey=True)    
    
    for ilabel,labelname in enumerate(all_labels):
        ifname = f"{fname_prefix}{labelname}.joblib"
        print(f"Loading {ifname}")
        data = joblib.load(ifname)
        for imodel,model in enumerate(models):
            all_trues,all_preds=[],[]
            for ifold in data:
                val_variants = data[ifold][models[1]]["validation"]["variants"]
                preds = data[ifold][model]["validation"]["preds"]
                trues = data[ifold][model]["validation"]["trues"]

                # standardize preds so it is always a 1D array of the dysfunctional pred prob
                if preds.ndim == 2 and preds.shape[1] == 2:
                    preds = preds[:, 1]
                    
                all_trues.extend(trues) # for later 
                all_preds.extend(preds)
                
                fpr, tpr, thresholds = skm.roc_curve(trues, preds)
                axroc[imodel][ilabel].plot(fpr,tpr,alpha=0.1)
                precision,recall, thresholds = skm.precision_recall_curve(trues, preds)
                axprc[imodel][ilabel].plot(recall,precision,alpha=0.1)
                
                if (model=="am"):
                    mcc = skm.matthews_corrcoef(trues,(preds>0.3402).astype(int))
                    brier = skm.brier_score_loss(trues,preds)
                    mcc_valid_results[model][ilabel].append(mcc)
                    brier_valid_results[model][ilabel].append(brier)                    
                else:
                    ### MCC max
                    thresholds = np.arange(0.1,0.9,0.01)
                    mccThresh =  [skm.matthews_corrcoef(trues, (preds > thresh).astype(int)) for thresh in thresholds]
                    maxMCC = max(mccThresh)
                    maxMCCThresh = thresholds[np.argmax(mccThresh)]
                    print("max mcc:",maxMCC)
                    print(maxMCCThresh)
                    mcc_valid_results[model][ilabel].append(maxMCC)
                    brier = skm.brier_score_loss(trues,preds)
                    brier_valid_results[model][ilabel].append(brier)                                        

            # roc, prc of all trues and preds 
            fpr, tpr, thresholds = skm.roc_curve(all_trues, all_preds)
            auroc = skm.roc_auc_score(all_trues,all_preds)                    
            axroc[imodel][ilabel].plot(fpr,tpr,label=f'{auroc:.2f}')
            axroc[imodel][ilabel].legend()
            
            precision,recall, thresholds = skm.precision_recall_curve(all_trues, all_preds)
            auprc = skm.average_precision_score(all_trues,all_preds)                    
            line=axprc[imodel][ilabel].plot(recall,precision,label=f'{auprc:.2f}')
            tpr_base = sum(all_trues)/len(all_trues)
            axprc[imodel][ilabel].axhline(y=tpr_base,color=line[0].get_color(),alpha=0.7,linestyle='--')
            axprc[imodel][ilabel].legend()

    # END OF     for ilabel,labelname in enumerate(all_labels):
            
    ##########
    ##### Plot metrics
    #generate_box_plot(all_labels,models,mcc_valid_results,ofname='mcc_box_rfc.pdf')
    print("MCC violin plot:")
    generate_mcc_violin_plot(all_labels,models,mcc_valid_results,ofname='mcc_violin_rfc.pdf')
    print("Brier score violin plot:")    
    generate_mcc_violin_plot(all_labels,models,brier_valid_results,ofname='brier_violin_rfc.pdf')        

    model1 = 'am'
    model2=['orig','origam','origammpnnesm']
    generate_diff_plots(all_labels,model1,model2,mcc_valid_results,ofname='mcc_diff_rfc.pdf')    

    for ilabel,labelname in enumerate(all_labels):
        axprc[0][ilabel].set_title(labelname)
    
    for ilabel,labelname in enumerate(all_labels):
        axroc[0][ilabel].set_title(labelname)        

    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for imodel, model in enumerate(models):
        axprc[imodel][0].set_ylabel(model)
        for ilabel, labelname in enumerate(all_labels):
            ax = axprc[imodel][ilabel]
            ax.set_ylim([0,1])
            ax.set_xlim([0,1])
            ax.legend(fontsize=12)            

            ax.set_yticks(yticks)
            ax.set_xticks(xticks)
                
            # Set y-ticks for first column
            if ilabel != 0:
                ax.set_yticklabels([])
    
            # Set x-ticks for last row
            if imodel != len(models) - 1:
                ax.set_xticklabels([])
                
    for imodel, model in enumerate(models):
        axroc[imodel][0].set_ylabel(model)
        for ilabel, labelname in enumerate(all_labels):
            ax = axroc[imodel][ilabel]
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7) #, label='Random (y=x)')             
            ax.set_ylim([0,1])
            ax.set_xlim([0,1])
            ax.legend(fontsize=12)

            ax.set_yticks(yticks)
            ax.set_xticks(xticks)
            
            # Set y-ticks for first column
            if ilabel != 0 :
                ax.set_yticklabels([])
     
            # Set x-ticks for last row
            if imodel != len(models) - 1:
                ax.set_xticklabels([])
    
                
    return 

##############################
def trainSingle(savedParams,labelName,ds,model_features,model_name=None): 
    '''
    Train a model on the dataset, save it. Assumes I am using excludeWT=True
    '''

    print(f"Load {savedParams} for {labelName} using {model_features}. Train and save to {model_name}.")

    if model_features=='orig':
        idx0,idx1=0,12
    elif model_features=='origam':
        idx0,idx1=0,13
    elif model_features=='origammpnnesm':
        idx0,idx1 = 0,15
    else:
        print(f"Unrecognized option for model_features: {model_features}. Exiting.")
        return

    model = model_features
    sorted_residues = sorted(ds.featureDict.keys())
    features_np = np.stack([ds.featureDict[name].detach().cpu().numpy() for name in sorted_residues])[:,idx0:idx1]
    labels_np = (np.stack([ds.labelDict[name].detach().cpu().numpy() for name in sorted_residues]) >= 1).astype(int)

    splits = StratifiedShuffleSplit(n_splits=100, test_size=0.2) #,random_state=14926)
    load_params = joblib.load(savedParams)
    saved_pipeline = load_params[labelName][model]
    rf = saved_pipeline.named_steps['randomforestclassifier']
    saved_hyperparams = rf.get_params()
    print(saved_hyperparams)
    best_model = make_pipeline(StandardScaler(),RandomForestClassifier(**saved_hyperparams))
    
    best_mcc = -1.0 # initialize, use max mcc 
    thresholds = np.arange(0.1,0.9,0.01)    
        
    for fold, (train_idx, test_idx) in enumerate(splits.split(features_np,labels_np)):
        X_train_orig , y_train_orig = features_np[train_idx], labels_np[train_idx]
        X_test_orig,y_test_orig = features_np[test_idx],labels_np[test_idx]
        
        X_train = X_train_orig
        y_train = y_train_orig

        # Fit on training features, test on test split
        perturb_mask = (X_test_orig[:, 0:4].sum(axis=1) != 0)    # only get perturbing variants for test
        X_test = X_test_orig[perturb_mask]
        y_test = y_test_orig[perturb_mask]
        valid_indices = [test_idx[i] for i in np.where(perturb_mask)[0]]
        valid_vars = [sorted_residues[i] for i in valid_indices]

        pipeline = best_model  # ['origammpnnesm']
        pipeline.fit(X_train[:,idx0:idx1],y_train)
        y_pred_proba = pipeline.predict_proba(X_test[:,idx0:idx1])

        mccThresh =  [skm.matthews_corrcoef(y_test, (y_pred_proba[:,1] > thresh).astype(int)) for thresh in thresholds]
        maxMCC = max(mccThresh)
        maxMCCThresh = thresholds[np.argmax(mccThresh)]
        print(f"  max MCC= {maxMCC}, thresh={maxMCCThresh}")
        
        if maxMCC > best_mcc:
            print("Updating saved model")
            best_mcc = maxMCC
            brier = skm.brier_score_loss(y_test,y_pred_proba[:,1])
            joblib.dump({'model': pipeline, 'thresh': maxMCCThresh, 'maxmcc': maxMCC, 'brier': brier}, model_name)

    # Save and output the best model
    print(f"Best MCC: {best_mcc:.4f} — model saved to {model_name}")    
        
    return 

##############################
def testSingle(ds,savedModel,labelname,model_features):
    ''' 
    Predict variables in ds using savedModel. 
    '''

    if model_features=='orig':
        idx0,idx1=0,12
    elif model_features=='origam':
        idx0,idx1=0,13
    elif model_features=='origammpnnesm':
        idx0,idx1 = 0,15
    else:
        print(f"Unrecognized option for model_features: {model_features}")
        return

    saved = joblib.load(savedModel)
    pipeline = saved['model']  # now pipeline is your actual sklearn pipeline
    thresh = saved['thresh']   # saved mcc cutoff threshold

    residues = ds.variantNames
    sorted_residues = sorted(residues, key=lambda x: int(''.join(filter(str.isdigit,x))))

    testPreds = {}    
    for name in sorted_residues:
        features_np = ds.featureDict[name].detach().cpu().numpy()
        output = pipeline.predict_proba(features_np[idx0:idx1].reshape(1,-1))
        pred = int(output[0,1]>thresh)

        key=labelname+"_"+name
        testPreds[key] = {
            "output": output,
            "preds": pred,
            "trues":  ds.labelDict[name].detach().cpu().numpy()
        }

    return testPreds 

##############################
def testSingleNoGTLabels(ds,savedModel,labelname,model_features):
    ''' 
    Predict variables in ds using savedModel. Data has no ground truth labels. 
    '''

    if model_features=='orig':
        idx0,idx1=0,12
    elif model_features=='origam':
        idx0,idx1=0,13
    elif model_features=='origammpnnesm':
        idx0,idx1 = 0,15
    else:
        print(f"Unrecognized option for model_features: {model_features}")
        return
    
    saved = joblib.load(savedModel)
    pipeline = saved['model']  # now pipeline is your actual sklearn pipeline
    thresh = saved['thresh']   # saved mcc cutoff threshold

    residues = ds.variantNames
    sorted_residues = sorted(residues, key=lambda x: int(''.join(filter(str.isdigit,x))))

    testPreds = {}    
    for name in sorted_residues:
        features_np = ds.featureDict[name].detach().cpu().numpy()
        output = pipeline.predict_proba(features_np[idx0:idx1].reshape(1,-1))
        pred = int(output[0,1]>thresh)

        key=labelname+"_"+name
        testPreds[key] = {
            "output": output,
            "preds": pred
            ####"trues":  ds.labelDict[name].detach().cpu().numpy()
        }
        
    return testPreds 

##############################
def getValListClass(ds,fname_prefix,labelname,check_list=None):
    """
    Gather the validation classification predictions per split. These will be saved for all the labels together. 
    """

    ifname = f"{fname_prefix}{labelname}.joblib"
    print(f"Loading {ifname}")
    data = joblib.load(ifname)
    #ifold=0

    model = 'origammpnnesm' 
    
    if check_list is not None: 
        checkListVars = np.loadtxt(check_list, dtype=str)

    pred_list = []
        
    for ifold in data:
        #print(ifold)
        
        valid_vars = data[ifold][model]["validation"]["variants"]
        maxMCCThresh = data[ifold][model]["mcc_thresh"]
        
        ## max mcc
        #thresholds = np.arange(0.1,0.9,0.01)
        #mccThresh =  [skm.matthews_corrcoef(data[ifold][model]["validation"]["trues"], (data[ifold][model]['validation']['preds'][:,1] > thresh).astype(int)) for thresh in thresholds]
        #maxMCC = max(mccThresh)
        #maxMCCThresh = thresholds[np.argmax(mccThresh)]
        #print(" max mcc, threshold=",maxMCC,maxMCCThresh)

        for i in range(len(valid_vars)):
            name=valid_vars[i]
            model_pred = data[ifold][model]['validation']["preds"][i,1]        

            if name in checkListVars:
                print(f"found {name}")
                pred_list.append([name,int((model_pred>maxMCCThresh).item())])
                

    votes = defaultdict(list)
    for var,pred in pred_list:
        votes[var].append(pred)

    valPreds = {}
    
    for var,preds in votes.items():
        majority = Counter(preds).most_common(1)[0][0]
        key=labelname+"_"+var
        #print(key)
        valPreds[key]= {
            #"variant": var,
            "preds": majority,
            "trues": ds.labelDict[var].item()
        }
                        
    return valPreds 

##############################
def add_missing_heatmap_cells(ds,savedModel,ax,cell,cmap):
    ''' 
    Add to ax heatmap at cell (i,j). 
    Color/label in cell is based on variant prediction for that label. 
    Use saved model in fname to predict. 
    Variant may be msising bc it is either not_measured (label=6) or "uncertain" (label=4).
    Color scheme and label should be the same as in plot_scripts::plotVarClass(). 
    '''
    
    i,j = cell["i"], cell["j"]
    variant,label = cell["variant"],cell["label"]
    
    # Load saved model , predict
    idx0,idx1 = 0,15 # all features
    saved = joblib.load(savedModel)
    pipeline = saved['model']  # now pipeline is your actual sklearn pipeline
    thresh = saved['thresh']   # optional, if you need threshold too

    features_np = ds.featureDict[variant].detach().cpu().numpy()    
    true = ds.labelDict[variant].detach().cpu().numpy()
    output = pipeline.predict_proba(features_np[idx0:idx1].reshape(1,-1))
    pred = int(output[0,1]>thresh)
    
    norm = plt.Normalize(vmin=0, vmax=1)  # needs to match the vmin/vmax in plotVarClass()
    color = cmap(norm(pred))

    rect = plt.Rectangle((j - 0.5, i - 0.5), 1,1,
                         color=color,
                         edgecolor="black",
                         linewidth=1.5,
                         zorder=2)
    ax.add_patch(rect)

    if true == 4:
        true_str = "UNC"
    elif true == 6:
        true_str = "NM"
    else:
        true_str = "UNK" # neither expected label. check. 

    ax.text(j, i, f"{pred}\n ({true_str})", ha='center', va='center', color='grey')
    
    return 

##############################
def cf_crossvalPreds(fname_prefix,models,check_list,featuresCSVList,trainLabelsCSV,config):
    ''' 
    Get prediction distributions (#pop pathogenic per metric), also 
    accuracy of lof, gof, and other var classes, or overall the loaded set.
    '''
    
    if check_list is not None: 
        checkListVars = set(np.loadtxt(check_list, dtype=str))
    else:
        checkListVars = None

    gof = defaultdict(lambda: defaultdict(list))
    lof = defaultdict(lambda: defaultdict(list))
    normal = defaultdict(lambda: defaultdict(list))

    all_labels = [cfg["name"] for cfg in config["labels"]]

    for labelname in all_labels:
        ifname = f"{fname_prefix}{labelname}.joblib"
        print(f"Loading {ifname}")
        data = joblib.load(ifname)
        # load variant data for this label
        labelspecs = next(labelspecs for labelspecs in config["labels"] if labelspecs["name"] == labelname)
        rcut, pvalcut, gofspec = labelspecs["rcut"], labelspecs["pvalcut"], labelspecs["gof"]
        ds = VariantDataset(featuresCSVList,trainLabelsCSV,labelname,rcut,pvalcut,gofspec,True,True,True,False,False,None,None)
        for model in models: 
            for ifold in data:
                val_variants = data[ifold][models[1]]["validation"]["variants"]  #variants list is the same in order for all saved models 
                #preds = (data[ifold][model]["validation"]["preds"] > 0.5).astype(int)
                preds = data[ifold][model]["validation"]["preds"]
                trues_binary = data[ifold][model]["validation"]["trues"]
                trues_label = np.array([ds.labelDict[var].detach().cpu().numpy() for var in val_variants])
                # standardize preds so it is always a 1D array of the dysfunctional pred prob
                if preds.ndim == 2 and preds.shape[1] == 2:
                    preds = preds[:, 1]
                for variant, true, pred in zip(val_variants, trues_label, preds):
                    if checkListVars is not None and variant not in checkListVars:
                        continue
                    if true==5: # gof
                        gof[labelname][model].append(pred)
                    elif true==1: # lof
                        lof[labelname][model].append(pred)                        
                    elif true==0: # normal
                        normal[labelname][model].append(pred)
    # end for loop

    ## plot comparison metrics 
    # scatter plot of brier scores. use same color settings as in generate_mcc_violin_plot
    fig,ax=plt.subplots(3,1,figsize=(7,4),sharey=True,sharex=True)
    xpos = np.arange(len(all_labels))    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    colors = [(r, g, b, 0.7) for r, g, b, _ in colors]  # replace original alpha
    width=0.11 # for 5 models (incl. mpnn) 
    gap=0.05

    for ilabel,labelname in enumerate(all_labels):
        for imodel,model in enumerate(normal[labelname]):
            preds = normal[labelname][model]
            if not preds:
                continue
            preds = np.array(preds)
            brier = np.mean((preds - 0)**2) # 0 is true label for normal
            print(f"{labelname} – {model}: NORMAL Brier score = {brier:.3f}")
            ax[0].scatter(xpos[ilabel]+imodel*(width+gap) , brier,color=colors[imodel],edgecolor='k')
            ax[0].set_ylabel("Normal")
            
    for ilabel,labelname in enumerate(all_labels):
        for imodel,model in enumerate(lof[labelname]):
            preds = lof[labelname][model]
            if not preds:
                continue
            preds = np.array(preds)
            brier = np.mean((preds - 1)**2) # 1 is binary true label for lof
            print(f"{labelname} – {model}: LOF Brier score = {brier:.3f}")
            ax[1].scatter(xpos[ilabel]+imodel*(width+gap) , brier,color=colors[imodel],edgecolor='k')
            ax[1].set_ylabel("LOF")
            
    for ilabel,labelname in enumerate(all_labels):
        for imodel,model in enumerate(gof[labelname]):
            preds = gof[labelname][model]
            if not preds:
                continue
            preds = np.array(preds)
            brier = np.mean((preds - 1)**2) # 1 is binary true label for gof 
            print(f"{labelname} – {model}: GOF Brier score = {brier:.3f}")
            ax[2].scatter(xpos[ilabel]+imodel*(width+gap) , brier,color=colors[imodel],edgecolor='k')
            ax[2].set_ylabel("GOF")

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])

    xcenter = [xpos[i] + (len(models)-1)*(width+gap)/2 for i in range(len(all_labels))]
    [a.set_xticks(xcenter) for a in ax]
    ax[2].set_xticklabels(all_labels)
            
    [a.set_yticks(np.arange(0,0.71,0.1)) for a in ax]            
    [a.tick_params(axis='both', labelsize=10) for a in ax]
    [a.grid(True,alpha=0.3) for a in ax]
    
    #plt.show()
    plt.savefig("brier_breakdown.png",bbox_inches="tight",dpi=300)
    
    return

##############################
def writePredstoHDF5(labels,testpreds,ofname):
    ''' Write testpreds to h5 file. This is only for output from noGTlabels as of now; true values are stored.'''

    variants0 = set(key.rsplit('_', 1)[1] for key in testpreds.keys())
    variants = sorted(variants0,key=getpos) 

    n_rows = len(variants)
    n_cols = len(labels)
    preds = np.zeros((n_rows, n_cols), dtype=np.int8)
    preds_proba = np.zeros((n_rows, n_cols), dtype=np.float32)

    # Build mapping from variant to row
    variant_idx = {v:i for i,v in enumerate(variants)}
    label_idx   = {l:i for i,l in enumerate(labels)}

    for key, d in testpreds.items():
        label , var = key.rsplit("_",1)
        i = variant_idx[var]
        j = label_idx[label]
        preds[i,j] = int(d["preds"])
        preds_proba[i,j] = float(d["output"][0,1])
        
    with h5py.File(ofname, "w") as f: # Write to HDF5
        dt_str = h5py.string_dtype("utf-8")
        f.create_dataset("variants", data=np.array(variants, dtype=dt_str))
        f.create_dataset("labels", data=np.array(labels, dtype=dt_str))
        f.create_dataset("preds", data=preds)
        f.create_dataset("preds_proba", data=preds_proba)
        
        # Variant index
        f.create_dataset("variant_index_keys", data=np.array(variants, dtype=dt_str))
        f.create_dataset("variant_index_values", data=np.arange(n_rows, dtype=np.int32))

    return

######################################################################
if __name__ == "__main__":
    main()
