''' Script to compare hyperparams from grid search CV, and also print feature importance for saved model.'''

import joblib
import numpy as np 

labels=['iks','v12','tau_act','tau_deact','cell_surf_exp','tot_exp','traffick_eff']

##############################
# Print the hyperparams:  

params =  joblib.load('out_rfc._hyperparams.joblib')
model = 'origammpnnesm'

for labelName in labels:
    print(labelName)
    pipeline = params[labelName][model]
    rf = pipeline.named_steps['randomforestclassifier']
    saved_hyperparams = rf.get_params()    
    for param in ['n_estimators', 'max_depth', 'max_features']:
        print("  " , param, ":", saved_hyperparams[param])

##############################
# Print feature importance:

model_prefix="save_model_state.rfc.biophys_evol_am_mpnnesm."

featuresList=['change no. H acceptor sites','change no. H donor sites','change volume of aa','change PSSM NR','mutat AA hydrophobicity','mutant AA polarizability','functional density (polarizability 6.5 A)','functional density (polarizability 12 A)','functional density (hydrophobicity 1 A)','functional density (hydrophobicity 6.5 A)','distance from channel pore axis','burial on membrane','AM','MPNN','ESM']
    
for labelName in labels:
    saved = joblib.load(model_prefix+labelName+".joblib")

    pipeline = saved['model'] 
    rf = pipeline.named_steps['randomforestclassifier']
    importances = rf.feature_importances_
    feature_names = featuresList
    top_indices = np.argsort(importances)[::-1][:5] #get top 5 
    
    print(f"Top 5 features for {labelName}:")
    for i, idx in enumerate(top_indices, start=1):
        print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")


# end
