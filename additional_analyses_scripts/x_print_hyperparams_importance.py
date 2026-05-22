#!/usr/bin/env python3

''' 
Print saved model hyperparameters, calc and plot global feature importance. 
Needs saved hyperparameters from the out_*hyperparams.joblib
Needs saved model save_model_..*.joblib 
'''

import joblib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

labels=['iks','v12','tau_act','tau_deact','cell_surf_exp','tot_exp','traffick_eff']

model_path="../saved_models/"

##############################
# Print the hyperparams:  

params =  joblib.load(model_path+'out_rfc._hyperparams.joblib')
model = 'origammpnnesm'

for labelName in labels:
    print(labelName)
    pipeline = params[labelName][model]
    rf = pipeline.named_steps['randomforestclassifier']
    saved_hyperparams = rf.get_params()    
    for param in ['n_estimators', 'max_depth', 'max_features']:
        print("  " , param, ":", saved_hyperparams[param])

##############################
# Print&plot  feature importance:

model_prefix=model_path+"trained_on_all/save_model_state.rfc.biophys_evol_am_mpnnesm."

featuresList=['Δ Num. H acceptor sites','Δ Num. H donor sites','Δ Volume AA','Δ PSSM NR','Mutant AA hydrophobicity','Mutant AA polarizability','Funct. density (polarizability 6.5 Å)','Funct. density (polarizability 12 Å)','Funct. density (hydrophobicity 1 Å)','Funct. density (hydrophobicity 6.5 Å)','Distance from channel pore axis','Burial in membrane','AM','ProteinMPNN','ESM']

top_feature_counts = Counter()
all_importances = []

for labelName in labels:
    saved = joblib.load(model_prefix+labelName+".joblib")

    pipeline = saved['model'] 
    rf = pipeline.named_steps['randomforestclassifier']
    importances = rf.feature_importances_
    feature_names = featuresList

    # store all importances
    all_importances.append(importances)
    
    # get top N features
    top_indices = np.argsort(importances)[::-1][:5]
    
    for i, idx in enumerate(top_indices, start=1):
        featname=feature_names[idx]
        print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")
        top_feature_counts[featname] += 1 # counter

# convert importances to np array for stats
all_importances = np.array(all_importances)
avg_importances = np.mean(all_importances, axis=0)
std_importances = np.std(all_importances, axis=0)

# sort
sorted_indices = np.argsort(avg_importances)[::-1]
sorted_features = np.array(feature_names)[sorted_indices]
sorted_avg = avg_importances[sorted_indices]
sorted_std = std_importances[sorted_indices]

# plot
plt.figure(figsize=(10, 5))
plt.barh(sorted_features[:15], sorted_avg[:15], xerr=sorted_std[:15],
         capsize=4,color='bisque',edgecolor='k')
plt.xlabel("Average Feature Importance",fontsize=12)
plt.gca().invert_yaxis()
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()
# plt.savefig("fig_avg_feat_importance.pdf",dpi=300)

