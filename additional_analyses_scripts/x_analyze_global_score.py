'''
Calculate global score from function metrics and analyze. 
All prediction outputs are saved to hdf5 file. 
Only call the variants I am interested in (specify in list).
'''

import numpy as np
import math
import h5py
import re 
import matplotlib.pyplot as plt
import seaborn as sbn
from itertools import combinations
from scipy.stats import mannwhitneyu,spearmanr
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import MultipleLocator
from Bio import PDB
from matplotlib.colors import LinearSegmentedColormap

######################################################################
def main():

    ifname="kcnq1_predictions.hdf5"

    ##########
    #ifname_varlists=["list_gnomad_likely_benign.txt","list_clinvar_benign.txt","list_clinvar_likely_benign.txt","list_clinvar_likely_pathogenic.txt","list_clinvar_pathogenic.txt"]
    ifname_varlists=["list_gnomad_gt_1e-4.txt","list_gnomad_gt_1e-5.txt","list_clinvar_benign.txt","list_clinvar_likely_benign.txt","list_clinvar_likely_pathogenic.txt","list_clinvar_pathogenic.txt"]

    #ifname_varlists=["list_clinvar_likely_benign.txt","list_clinvar_benign.txt"]
    #ifname_varlists=["list_clinvar_likely_pathogenic.txt","list_clinvar_pathogenic.txt"]
    #ifname_varlists=["list_gnomad_gt_1e-4.txt"]
    #ifname_varlists=["list_gnomad_gt_1e-5.txt"]

    #ifname_varlists=["list_unc.txt"]
    ## ifname_varlists=["all_am_q1_ambiguous.txt"]
    #ifname_varlists=["list_twist_250929.txt"]
    ##########

    ##########    
    # hard-set weights 
    ## for function
    weights = [0.36,0.2,0.23,0.21] # see x_get_global_score_weights: iks, v12, tau_act, tau_deact ; testing different weights I consistently get ~0.45
    omin,omax=0,4
    ### for trafficking    
    #weights = [0.33,0.33,0.33]
    #omin,omax=4,7
    ##########
    
    ## Get global scores
    globalScoreDict = {}
    for f in ifname_varlists:
        listname = f.split("list_")[1].split(".txt")[0]
        includeVars = np.loadtxt(f,dtype=str)
        variants, preds = readHDF5(ifname,includeVars)
        globalScore_func = np.sum(preds[:,omin:omax] * weights,axis=1)
        globalScoreDict[listname] = {"variants":variants,"globalScore_func":globalScore_func}

    ##########
    ### For ClinVar+gnomad lists :
    #printListinPlot(globalScoreDict)
    calcCorr(globalScoreDict)    # quantify relationship with correlation calc
    plotDist(globalScoreDict)
    plotBarStats(globalScoreDict)
    plt.show()
    ############    
    
    ##### structure visualization
    #genPDBcolorbyscore(globalScoreDict,
    #                   "/home/changga/Documents/kcnq1e1/paper_2508/inputs/8sik_modeller_trunc.pdb",
    #                   "out_kcnq1_colorbyscore.pdb"
    #                   )
    #renderColorLegend() # input should match with palette used in ~/Documents/kcnq1e1/paper_2508/fig_panels/view_globalScore.py
    
    return 

##############################
def readHDF5(ifname,selVars):
    ''' Read saved predictions and return variants and 
    preds matching selected list. Prints warning if variant not found.
    '''

    # when i need a column, can do j = np.where(labels == "v12")[0][0]
    
    with h5py.File(ifname,"r") as f:
        labels = f["labels"][:].astype(str) 
        variants = f["variants"][:].astype(str)
        preds = f["preds"][:]
        
        # build variant index
        variant_index = dict(zip(f["variant_index_keys"][:].astype(str),
                                 f["variant_index_values"][:]))

    # Map subset variants to row indices, skip missing
    rows = [variant_index[v] for v in selVars if v in variant_index]
    foundVars =  variants[rows]
    foundPreds = preds[rows, :]
    ## sort by residue number 
    sort_idx = sorted(range(len(foundVars)), key=lambda i: getpos(foundVars[i]))
    foundVars = foundVars[sort_idx]
    foundPreds = foundPreds[sort_idx,:]

    print(f"Loading {len(foundVars)} variants.")
    
    # Check for skipped variants
    skipped = set(selVars) - set(foundVars)
    if skipped:
        print(f"Warning: {len(skipped)} variant(s) not found in HDF5 and will be skipped: {skipped}")
    
    return foundVars,foundPreds

##############################
def plotDist(scoresDict):
    ''' Plot scoresDict distribution per variant list set.''' 

    fig,ax = plt.subplots(figsize=(10,4))

    labels = list(scoresDict.keys())
    data = [v["globalScore_func"] for v in scoresDict.values() if len(v["globalScore_func"]) > 0]

    sbn.violinplot(data=data, ax=ax,inner="box",cut=0)
    #sbn.boxplot(data=data,ax=ax)
    sbn.swarmplot(data=data,ax=ax) 

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.tick_params(axis='y',labelsize=12)
    ax.tick_params(axis='x',labelsize=12)    
    ax.grid(True,alpha=0.3)

    plt.tight_layout()
    
    return

##############################
def plotBarStats(scoresDict):
    ''' Bar plot, Mann Whitney for stat comparison.'''

    labels = list(scoresDict.keys())
    means = [np.mean(v["globalScore_func"]) for v in scoresDict.values()]
    stds  = [np.std(v["globalScore_func"], ddof=1) for v in scoresDict.values()]
    x = np.arange(len(labels))

    # Plot mean ± SD bar plot with horizontal caps
    fig, ax = plt.subplots(figsize=(3.5,4))
    ax.bar(
        x, means,
        yerr=stds,
        width=0.4,
        capsize=3,                # horizontal cap length
        #color='tab:blue',
        color='bisque',
        edgecolor='k',
        #ecolor='k',
        error_kw={'elinewidth':1, 'capthick':0, 'alpha':0.9}
    )

    # jitter individual data 
    for i, label in enumerate(labels):
        data = scoresDict[label]["globalScore_func"]
        jitter = np.random.uniform(-0.25, 0.25, size=len(data))  # horizontal jitter
        ax.scatter(
            np.full_like(data, x[i]) + jitter,
            data,
            facecolors='white',
            edgecolors='k',
            linewidths=0.8,
            alpha=0.7,
            s=12,
            zorder=3
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([-0.015,1.017])
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    #for spine in ax.spines.values():
    #    spine.set_alpha(0.5)
    #for spine in ax.spines.values():
    #    spine.set_visible(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)

    ax.tick_params(axis='y',labelsize=12)
    ax.set_ylabel("Global score of X")
    #ax.set_title("Mean ± SD Global Scores per Category")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('output_bar.pdf',format='pdf',bbox_inches="tight")        

    
    # Mann-Whitney pairwise tests
    # Prepare pairwise comparisons
    group_values = [scoresDict[k]['globalScore_func'] for k in labels]
    pairs = list(combinations(range(len(labels)), 2))
    raw_pvals = []
    
    # Compute raw p-values
    for i,j in pairs:
        stat, p = mannwhitneyu(group_values[i], group_values[j], alternative='two-sided')
        raw_pvals.append(p)
        
    # Multiple testing correction
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method='bonferroni')
    
    # Print automatic summary lines
    print("Pairwise Mann-Whitney U tests with Bonferroni correction:")
    for k, (i,j) in enumerate(pairs):
        grp1, grp2 = labels[i], labels[j]
        mean1, mean2 = np.mean(group_values[i]), np.mean(group_values[j])
        if mean1!=0 and mean2!=0:
            log2fc = math.log2(mean1/mean2)
        else:
            log2fc = np.nan
        d = cohen_d(group_values[i], group_values[j])
        p_text = "<0.05" if pvals_corrected[k] < 0.05 else f"{pvals_corrected[k]:.2f}"
        significant = "Yes" if reject[k] else "No"
        summary_line = (
            f"{grp1} vs {grp2}: log2FC={log2fc:.2f}, Cohen's d={d:.2f}, "
            f"p={p_text}, significant={significant}"
        )
        print(summary_line)
        
    return

##############################
def calcCorr(scoresDict):
    ''' Calculate correlation b/en scores and categories.'''
    
    # Hard-set arbitrary values for all the variants of a category
    assigned_values_map = { # spearman rank only cares about order 
        'gnomad_likely_benign': 0, #1,
        'gnomad_gt_1e-4': 0, #1,
        'gnomad_gt_1e-5': 0,
        'clinvar_benign': 0,
        'clinvar_likely_benign': 0, #1,
        'clinvar_likely_pathogenic': 1,        
        'clinvar_pathogenic': 1 #3
    }

    # Flatten all global scores and assigned values
    all_scores, all_assigned = [],[]
    
    for group_name, group_data in scoresDict.items():
        n = len(group_data['variants'])
        all_scores.extend(group_data['globalScore_func'])
        all_assigned.extend([assigned_values_map[group_name]] * n)

    all_scores = np.array(all_scores)
    all_assigned = np.array(all_assigned)

    corr, pval = spearmanr(all_scores, all_assigned)

    print(f"Spearman rank correlation: rho = {corr:.3f}, p = {pval:.3e}")
    
    plt.scatter(all_assigned, all_scores, color='blue') # quick plot to visualize
    
    return 

##############################
def genPDBcolorbyscore(scoresDict,inpdb,outpdb):
    ''' Generate a PDB where the glboal score is in the B-factor so I can 
    color by the pdb residues by score. ''' 

    residue_scores = {}
    for group in scoresDict.values():
        for variant, score in zip(group['variants'], group['globalScore_func']):
            match = re.search(r'([A-Z])(\d+)([A-Z])', variant)
            if match:
                resnum = int(match.group(2))
                if resnum in residue_scores:
                    print(f"WARNING: Duplicate residue {resnum}: overwriting previous score {residue_scores[resnum]:.3f} with {score:.3f}")
                residue_scores[resnum] = score

    ## read structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", inpdb)

    ## assign b factor to residue 
    assigned_residues = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                resnum = residue.get_id()[1]
                score = residue_scores.get(resnum, 2.0)  # dummy for variants not in list
                for atom in residue:
                    atom.set_bfactor(score)

    ## save to pdb
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(outpdb)

    print(f"Saving PDB where bfactor=score to {outpdb}. Residues not in list have bfactor=2")
    
    return

##############################
def renderColorLegend():

    #smin,smax = 0,1
    #cmap = plt.get_cmap(palette)
    ## Create a gradient
    #gradient = np.linspace(smin, smax, 256).reshape(1, -1)
    #
    #fig, ax = plt.subplots(figsize=(3, 1))
    #ax.imshow(gradient, aspect='auto', cmap=cmap)
    #ax.set_yticks([])
    #
    ## Add labels at fixed intervals
    #interval = 0.2  # label interval
    #ticks = np.arange(smin, smax + interval, interval)
    #tick_pos = ((ticks - smin) / (smax - smin)) * 255  # map to pixel positions
    #ax.set_xticks(tick_pos)
    #ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    #ax.xaxis.set_ticks_position('bottom')
    #ax.tick_params(axis='x', length=6, width=1, labelsize=12)
    #
    #plt.tight_layout()
    #plt.show()

    colors = ["white", "purple"]
    cmap = LinearSegmentedColormap.from_list("white_purple", colors)
    
    # Create a figure with a colorbar
    fig, ax = plt.subplots(figsize=(1, 4))  # vertical colorbar
    norm = plt.Normalize(vmin=0, vmax=1)     # adjust min/max to your data range
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    plt.show()
    

    return

##############################
def printListinPlot(scoresDict):
    ''' Plot vars in y vs global score in x to quickly look at scores.'''

    print(scoresDict)
    
    fig,ax=plt.subplots(figsize=(3,7.5))
    for labels,vars in scoresDict.items():
        variants = vars['variants']
        scores = vars['globalScore_func']
        ax.scatter(scores,variants,color='bisque',edgecolor='k')

    ax.invert_yaxis()
    ax.grid(True, which='both',alpha=0.1)
    ax.tick_params(axis='both', which='both', labelsize=12)
    ax.set_xlim([-0.03,1.03])
    ax.xaxis.set_minor_locator(MultipleLocator(0.1)) # 0.5)) #1.0))
    ax.xaxis.set_major_locator(MultipleLocator(0.2)) # 0.5)) #1.0))    

    plt.tight_layout()
    plt.savefig('output_print_list_scores.pdf',format='pdf',bbox_inches="tight")        
    return 

##############################
def getpos(variant):
    match = re.match(r'([A-Za-z])(\d+)([A-Za-z])', variant)
    if match:
        start, num, end = match.groups()
        return (int(num), end)   # sort by number first, then final letter
    return (float('inf'), "")

##############################
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

######################################################################
if __name__ == "__main__":
    main()
