#!/usr/bin/env python3

''' 
Plot structure-based features for the cryoEM models. 
'''

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter

##########
all_features=['change no. H acceptor sites','change no. H donor sites','change volume of aa','change PSSM NR','mutat AA hydrophobicity','mutant AA polarizability','functional density (polarizability 6.5 A)','functional density (polarizability 12 A)','functional density (hydrophobicity 1 A)','functional density (hydrophobicity 6.5 A)','distance from channel pore axis','burial in membrane','AM','MPNN','ESM']

# structure ("positional") features do not change per variant 
struct_features= all_features[6:12] + list([all_features[13]]) # structural feature indices from the full set !!! for pdb indexing look below!!! 

# features from six cryoEM models
featuresCSVList =  [
    "../multimodel_features_yaml/individual_pdb_features/features_8sin_modeller_trunc.w_am_mpnn_esm.csv",
    "../multimodel_features_yaml/individual_pdb_features/features_8sim_modeller_trunc.w_am_mpnn_esm.csv",
    "../features_8sik_clean.w_am_mpnn_esm.csv",
    "../multimodel_features_yaml/individual_pdb_features/features_7xni_modeller_trunc.w_am_mpnn_esm.csv",
    "../multimodel_features_yaml/individual_pdb_features/features_7xnk_modeller_trunc.w_am_mpnn_esm.csv",
    "../multimodel_features_yaml/individual_pdb_features/features_7xnl_modeller_trunc.w_am_mpnn_esm.csv"
]
pdbs=['8SIN','8SIM','8SIK','7XNI','7XNK','7XNL'] # order should match featuresCSVList

##########

featureDict = {}

for i,featurecsv in enumerate(featuresCSVList):
    print("Processing:", featurecsv)    
    fname = pdbs[i]
    # sanity check pdb matches with features list
    assert(fname.casefold() in (Path(featurecsv).stem).casefold())
    featureDict[fname] = {}
    with open(featurecsv, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            var = parts[0].strip()
            featureDict[fname][var] = parts[7:13] + [parts[14]]

# filter_resis
nonperturb_sample = {
    k:v
    for k, v in featureDict['8SIK'].items()
    if k[0] == k[-1]
}

nfeat=len(struct_features)
fnames = list(featureDict.keys())

##############################
## features plot -  used to generate plot_cryoem_struct_features.pdf
residues = list(nonperturb_sample.keys())
res_nums = [int(r[1:-1]) for r in residues]

fig,ax = plt.subplots(nfeat,figsize=(12,10))

for j in range(nfeat):
    vals = np.array([
        [float(featureDict[fname][res][j]) for fname in fnames]
        for res in residues
    ])

    means = vals.mean(axis=1)
    stds = vals.std(axis=1)

    ax[j].errorbar(
        #range(len(residues)),
        res_nums,
        means,
        yerr=stds,
        fmt='o',
        markersize=3,
        markerfacecolor='white',
        markeredgecolor='black',
        markeredgewidth=1,
        ecolor='k'
    )
    
for j in range(nfeat):
    ax[j].set_ylabel(struct_features[j])
    ax[j].set_xticks(range(min(res_nums), max(res_nums)+1, 20))
    
for a in ax[:-1]:
    a.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax[-1].set_xlabel("Residue position")

# manual y lim adjustment to match feat_by_class violin plot
ax[1].yaxis.set_major_locator(MultipleLocator(0.05))
ax[2].yaxis.set_major_locator(MultipleLocator(0.3))

plt.tight_layout()
# plt.savefig('plot_cryoem_struct_features.pdf',dpi=300)
plt.show()

##############################
# Generate mean, std, and max dev from 8sik from struct features for model training

residues = list(featureDict['8SIK'].keys())
combine = [] # to save dataframe

for res in residues:
    row = {"residue": res}

    for j in range(nfeat):
        vals = np.array([float(featureDict[f][res][j]) for f in fnames])
        static_val = float(featureDict['8SIK'][res][j])

        row[f"mean_{j}"] = vals.mean()
        row[f"std_{j}"] = vals.std()
        row[f"maxdev_{j}"] = np.max(np.abs(vals - static_val))

    combine.append(row)

df = pd.DataFrame(combine)
df.to_csv("stats_struct_features.csv", index=False, float_format="%.5f")

#####
# reduce resi list :
df_filtered = df[df["residue"].isin(nonperturb_sample.keys())].copy()
# plot to look at std, mean spread
std_cols = [c for c in df_filtered.columns if c.startswith("std_")]
df_filtered_std = df_filtered.melt(id_vars="residue", value_vars=std_cols,
                 var_name="feature", value_name="std")

# maxdev
maxdev_cols = [c for c in df_filtered.columns if c.startswith("maxdev_")]
df_filtered_maxdev = df_filtered.melt(id_vars="residue", value_vars=maxdev_cols,
                    var_name="feature", value_name="maxdev")

# STD heatmap
std_matrix = df_filtered[std_cols].values
plt.figure()
plt.imshow(std_matrix, aspect='auto')
plt.colorbar(label="STD")
plt.yticks(range(len(df_filtered)), df_filtered["residue"])
plt.title("STD Heatmap")
plt.show()

# MAXDEV heatmap
maxdev_matrix = df_filtered[maxdev_cols].values
plt.figure()
plt.imshow(maxdev_matrix, aspect='auto')
plt.colorbar(label="Max Deviation")
plt.yticks(range(len(df_filtered)), df_filtered["residue"])
plt.title("Max Deviation Heatmap")
plt.show()
