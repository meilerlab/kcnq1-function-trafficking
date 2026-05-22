import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn 
import re
import math

from scipy.stats import wilcoxon 
import sklearn.metrics as skm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
from collections import defaultdict
import joblib
from matplotlib_venn import venn3
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from variant_dataset import VariantDataset

plt.rcParams['figure.dpi'] = 300
##############################
def generate_box_plot(labels,models,values,ofname):
    
    # Bar plot 
    width=0.11 # for 5 models (incl. mpnn) 
    gap=0.05
    #width=0.15 # for 3 models
    #gap=0.05

    xpos = np.arange(len(labels))    
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    #colors = ['lightsalmon','lightskyblue','thistle','tab:red','tab:green']
    figbox,axbox = plt.subplots(figsize=(10,5)) # ephys 4 measurements
    #figbox,axbox = plt.subplots(figsize=(5.5,3)) # traffick 3 
    for idx,model in enumerate(models):
        data = [values[model][i] for i in range(len(labels))]
        bp = axbox.boxplot(
            data,
            positions=xpos + idx * (width+gap),
            widths=width,
            patch_artist=True,  # Fill boxes with color
            boxprops=dict(facecolor='none', color=colors[idx],linewidth=1.3,alpha=1,zorder=1),            
            #boxprops=dict(facecolor=colors[idx], color="black",alpha=0.3,zorder=1),
            #boxprops=dict(facecolor='none', color="black"), #,alpha=0.4),
            #capprops=dict(color="black"),
            capprops=dict(color=colors[idx]),
            #whiskerprops=dict(color="black"),
            whiskerprops=dict(linestyle='-',color=colors[idx]),
            #medianprops=dict(color="dodgerblue",linewidth=2),
            medianprops=dict(color=colors[idx],linewidth=2.5),
            #flierprops=dict(marker="o", markersize=5, color="black", alpha=0.5)  # Outliers
            flierprops=dict(marker="o", markersize=5, color=colors[idx], alpha=0.8)  # Outliers
        )
        
        #axbox.scatter(xpos + idx * (width+gap),means, c='lightsteelblue',marker='d',zorder=3,edgecolor='k',alpha=0.8,label='mean')

    for idx,model in enumerate(models):
        data = [values[model][i] for i in range(len(labels))]
        xarr= xpos+idx*(width+gap)
        #axbox.scatter(xpos+idx*(width+gap),[ival for ival in data],marker='o',edgecolor='k')
        axbox.scatter([xi for xi, yi in zip(xarr, data) for _ in yi], [v for yi in data for v in yi],
                      color=colors[idx],alpha=0.5,
                      edgecolor='k',zorder=5)

    axbox.set_xticks(xpos + (width + gap) * (len(models) - 1) / 2)
    axbox.set_xticklabels(labels)
    axbox.set_ylabel("validation performance metric")
    axbox.set_yticks(np.arange(0, 1.1, 0.1))  # Adjust the range as needed
    axbox.grid(True,alpha=0.2)
    #axbox.legend()

    plt.savefig(ofname, bbox_inches="tight")  # Perfect for papers

    return 

##############################
def generate_mcc_violin_plot(all_labels,all_models,values,ofname):
    
    width=0.11 # for 5 models (incl. mpnn) 
    gap=0.05
    #width=0.15 # for 3 models
    #gap=0.05

    xpos = np.arange(len(all_labels))    

    # color palette for mcc plots fig 2 
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))

    ## for the 3-model struct tests 
    #c0 = plt.cm.tab10(9)
    #c_rest = plt.cm.tab20b(np.linspace(0, 1, len(all_models)-1))
    #colors = np.vstack([c0, c_rest])

    colors = [(r, g, b, 0.7) for r, g, b, _ in colors]  # replace original alpha    
    
    #colors = ['lightsalmon','lightskyblue','thistle','tab:red','tab:green']
    data,labels,models = [],[],[]
    for model in all_models:
        for label_idx ,label in enumerate(all_labels):
            for vi in values[model][label_idx]:
                data.append(vi)
                labels.append(label)
                models.append(model)

    #plt.figure(figsize=(16,4)) # for mcc  , pretrained
    plt.figure(figsize=(14,4)) # for mcc
    #plt.figure(figsize=(14,1.8)) # for brier
    axviolin = sbn.violinplot(
        x=labels,
        y=data,
        hue=models,
        inner="box",
        palette=colors,
        cut=0,
        #linewidth=2.5,   # thicker line
        scale="width"
        #box_width=1        
    )

    axviolin.set_ylim(-0.21, 1.01) # for mcc
    axviolin.set_yticks(np.arange(-0.2, 1.01, 0.1)) # for mcc

    #axviolin.set_ylim(0,0.51) # for brier
    #axviolin.set_yticks(np.arange(0,0.51,0.1)) # for brier
    
    axviolin.tick_params(axis='y',labelsize=12)
    axviolin.tick_params(axis='x',labelsize=12)    
    
    axviolin.grid(True,alpha=0.3)
    axviolin.legend(loc='lower left')

    plt.savefig(ofname, bbox_inches="tight")  

    #####
    # Analyze stdev
    grouped = defaultdict(list)  # keys: (label, model), values: list of values
    for v, lbl, mdl in zip(data, labels, models):
        grouped[(lbl, mdl)].append(v)
    stats = {}
    for key, vals in grouped.items():
        vals_arr = np.array(vals)
        mean_val = vals_arr.mean()
        std_val = vals_arr.std(ddof=1)
        cv_val = std_val / mean_val if mean_val != 0 else np.nan
        stats[key] = (mean_val, std_val, cv_val)
        
    # To compare per label across models:
    labels_unique = sorted(set(labels))
    models_unique = sorted(set(models))

    for lbl in labels_unique:
        print(f"Label: {lbl}")
        for mdl in models_unique:
            if (lbl, mdl) in stats:
                mean_val, std_val, cv_val = stats[(lbl, mdl)]
                print(f"  Model: {mdl}  Mean: {mean_val:.4f}  Std: {std_val:.4f}  CV: {cv_val:.4f}")
            else:
                print(f"  Model: {mdl}  No data")
        print()

    # Setup colors for models
    colors_model = plt.cm.tab10(np.linspace(0, 1, len(models_unique)))
    model_to_color = {m: c for m, c in zip(models_unique, colors_model)}
    
    # Prepare data for plotting
    means, stds, cvs, cs, lbls, mdls = [], [], [], [], [], []
    for (lbl, mdl), (mean_val, std_val, cv_val) in stats.items():
        means.append(mean_val)
        stds.append(std_val)
        cvs.append(cv_val)
        cs.append(model_to_color[mdl])
        lbls.append(lbl)
        mdls.append(mdl)

    figstd, axstd = plt.subplots(figsize=(7, 5))
    axstd.scatter(means, stds, c=cs, marker='o', s=70,edgecolors='k')
    axstd.set_xlabel('Mean')
    axstd.set_ylabel('Standard Deviation')
    axstd.tick_params(axis='both',labelsize=12)
    
    ## Plot CV on right y-axis
    #axstd[1].scatter(means, cvs, c=cs, marker='o', s=70, edgecolors='k')
    #axstd[1].set_xlabel('Mean')    
    #axstd[1].set_ylabel('Coefficient of Variation')
    #_ = [ax.tick_params(axis='both', labelsize=12) for ax in (axstd[0], axstd[1])]

    # Add legend for models
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=mdl,
                              markerfacecolor=model_to_color[mdl], markersize=10)
                       for mdl in models_unique]
    axstd.legend(handles=legend_elements) # , bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate points with label and model (optional, comment out if too cluttered)
    for i in range(len(means)):
        axstd.annotate(f"{lbls[i]}", (means[i], stds[i]), textcoords="offset points",
                       xytext=(5, 5), ha='left', fontsize=8, color=cs[i])
        
    figstd.suptitle(f"Mean vs STD and CV (colored by Model) for {ofname.split('_')[0]}")
    figstd.tight_layout()
                
    return 

##############################
def generate_diff_plots(all_labels,model1,model2,values,ofname):
    ## Get MCC difference, get statistical significance
    
    # Collect data
    diffs,labels,models = [],[],[]
    grouped_diffs = []  # Store per group diffs for testing
    group_keys = []     # Store model-label keys for each group
    for jmodel in model2:
        for i, label in enumerate(all_labels):
            data1 = values[model1][i]
            data2 = values[jmodel][i]
            group_diff = [d1 - d2 for d1, d2 in zip(data1, data2)]
            diffs.extend(group_diff)
            labels.extend([label] * len(group_diff))
            models.extend([jmodel] * len(group_diff))
            grouped_diffs.append(group_diff)
            group_keys.append((label, jmodel))
            
    #palette = sbn.color_palette("muted", len(set(models)))
    palette = sbn.color_palette("pastel", len(set(models)))    
    palette = [(r, g, b, 0.7) for r, g, b in palette]
    
    # Create the figure
    plt.figure(figsize=(14, 3.5))
    axviolin = sbn.violinplot(
        x=labels,
        y=diffs,
        hue=models,
        palette=palette,
        cut=0,
        #width=0.4,
        #linewidth=1.5,
        scale="width",
        inner="box",
        #box_width=1 # .5
        #dodge=True
    )
    
    # Add a horizontal zero line
    plt.axhline(0, linestyle="--", color="red", alpha=0.6, linewidth=1)

    axviolin.tick_params(axis='y',labelsize=12)
    axviolin.tick_params(axis='x',labelsize=12)    
    axviolin.set_yticks(np.arange(-0.7, 0.4, 0.1))    
    axviolin.grid(True,alpha=0.3)
    
    plt.ylabel(f"{model1} - Other Models MCC Difference", fontsize=13)
    plt.title("Model Comparison Across Variant Classes", fontsize=15, weight="bold")
    plt.legend()

    # Perform Wilcoxon tests (one-sided: model1 < modelX)
    print("Wilcoxon test (model1 < modelX) p-values:")
    for (label, model), diff in zip(group_keys, grouped_diffs):
        try:
            stat, pval = wilcoxon(diff, alternative='less')
            print(f"  {model1} vs {model} ({label}): p = {pval:.4g}")
        except ValueError as e:
            print(f"  {model1} vs {model} ({label}): test failed ({e})")

    plt.savefig(ofname, bbox_inches="tight")  
        
    return 

##############################
def plotVarClass(labels,valpreds,cmap):
    '''
    Plot heatmap with class predictions. 
    '''

    variants = set(key.rsplit('_', 1)[1] for key in valpreds.keys())
    missing_cells = []
    
    rows = sorted(variants,key=getpos)        
    cols = labels    
    
    # initialize matrices
    data = np.full((len(rows),len(cols)),np.nan)
    data_proba = np.full((len(rows),len(cols)),np.nan)    
    annot = np.full((len(rows),len(cols)),"",dtype=object)
    
    for i , var in enumerate(rows):
        for j,label in enumerate(cols):
            key = f"{label}_{var}"
            if key in valpreds:
                pred, true,pred_proba = valpreds[key]["preds"], valpreds[key]["trues"],valpreds[key]["output"][0,1]
                data[i,j] = pred
                data_proba[i,j] = pred_proba
                annot[i,j] = f"{pred} ({true})"
                # if unc(true=4) or nm (true=6), mark accordingly                
                if true==4:
                    annot[i,j] = f"{pred} (UNC)"                    
                if true==6:
                    annot[i,j] = f"{pred} (NM)"                                                            
            else:
                missing_cells.append({
                    "variant": var,
                    "label": label,
                    "i": i,
                    "j": j
                })

    # plot heatmap
    #fig,ax=plt.subplots(figsize=(9,6))
    #fig,ax=plt.subplots(figsize=(9,2)) # to shorten for the ambiguous vars plot
    #fig,ax=plt.subplots(figsize=(9,8)) # to lengthen for TWIST 
    fig,ax=plt.subplots(figsize=(9,1.2)) # to shorten for twist codon change

    im = ax.imshow(data,cmap=cmap,vmin=0,vmax=1,aspect='auto')

    #for i in range(len(rows)):
    #    for j in range(len(cols)):
    #        if annot[i,j]:
    #            sdum_pred, sdum_true = annot[i,j].strip(")").split("(")
    #            pred,true = int(sdum_pred),int(sdum_true)
    #            color='k' if pred==int(true>=1) else 'r'
    #            ax.text(j,i,annot[i,j],ha='center',va='center',color=color,fontsize=10)

    # Initialize per-column lists
    all_preds = {j: [] for j in range(len(cols))}
    all_preds_proba = {j: [] for j in range(len(cols))}
    all_trues = {j: [] for j in range(len(cols))}

    for i in range(len(rows)):
        for j in range(len(cols)):
            if annot[i, j]:
                sdum_pred, sdum_true = annot[i, j].strip(")").split("(")
                if sdum_true=="UNC" or sdum_true=="NM":
                    ax.text(j, i, annot[i, j], ha='center', va='center', color='gray', fontsize=10)
                    continue 
                pred, true = int(sdum_pred), int(sdum_true)

                # Save for MCC calc
                all_preds[j].append(pred)
                all_trues[j].append(int(true >= 1))  # binarize truth
                
                # Annotate
                color = 'k' if pred == int(true >= 1) else 'r'
                ax.text(j, i, annot[i, j], ha='center', va='center', color=color, fontsize=10)

                # Save pred_proba
                all_preds_proba[j].append(data_proba[i,j])

    # Calculate and print MCC per column
    for j in range(len(cols)):
        if all_preds[j]:  # avoid empty columns
            mcc = skm.matthews_corrcoef(all_trues[j], all_preds[j])
            brier = skm.brier_score_loss(all_trues[j], all_preds_proba[j])
            auprc = skm.average_precision_score(all_trues[j],all_preds_proba[j])
            if len(set(all_trues[j])) > 1: 
                auroc = skm.roc_auc_score(all_trues[j],all_preds[j])
            else:
                auroc = -1
            print(f"Column {cols[j]} MCC: {mcc:.3f} Brier: {brier:.3f} AUROC: {auroc:.3f} AUPRC: {auprc:.3f} ")
                
    # ticks and labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)

    ax.xaxis.tick_top()
    ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
    ax.tick_params(axis='y', labelsize=12)

    #for i in range(data.shape[0] + 1):
    #    ax.axhline(i - 0.5, color='ghostwhite', linewidth=0.5)
    #for j in range(data.shape[1] + 1):
    #    ax.axvline(j - 0.5, color='ghostwhite', linewidth=0.5)
    
    # colorbar
    cbar = plt.colorbar(im,ax=ax,ticks=[0.25,0.75],shrink=0.5,pad=0.02)
    cbar.ax.yaxis.set_tick_params(pad=11)  # increase pad from bar to labels
    cbar.ax.tick_params(size=0) 
    cbar.ax.set_yticklabels(['Normal=0','Dysfunctional≥1'],rotation=90)
    # Center the labels horizontally
    for label in cbar.ax.get_yticklabels():
        label.set_ha('center')  # horizontally center
        label.set_va('center')  # vertically center
    
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Predicted class',labelpad=10,fontsize=12)
    
    #plt.tight_layout()
    plt.title(f"Test set predictions. pred(true)")
    plt.savefig('output_enable_test_heatmap.pdf',format='pdf',bbox_inches="tight")

    return ax,missing_cells

###############################
def getpos(variant):
    match = re.match(r'([A-Za-z])(\d+)([A-Za-z])', variant)
    if match:
        start, num, end = match.groups()
        return (int(num), end)   # sort by number first, then final letter
    return (float('inf'), "")

##############################
def plotTestPreds(labels,testpreds,cmap):
    ''' Plot test predictions for analysis. ''' 

    variants = set(key.rsplit('_', 1)[1] for key in testpreds.keys())
    
    rows = sorted(variants,key=getpos)        
    cols = labels    
    
    # initialize matrices
    data = np.full((len(rows),len(cols)),np.nan)
    annot = np.full((len(rows),len(cols)),"",dtype=object)
    
    for i , var in enumerate(rows):
        for j,label in enumerate(cols):
            key = f"{label}_{var}"
            pred = testpreds[key]["preds"]
            pred_proba = testpreds[key]["output"][0,1]
            data[i,j] = pred
            annot[i,j] = f"{pred_proba:.2f}"

    label_pred_counts = np.sum(data, axis=0).astype(int)  # shape: (n_labels,)
    print(label_pred_counts)
            
    # plot heatmap
    fig,ax=plt.subplots(figsize=(9,6))
    im = ax.imshow(data,cmap=cmap,vmin=0,vmax=1,aspect='auto')

    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, annot[i, j], ha='center', va='center', color='k', fontsize=10)

    # ticks and labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)

    ax.xaxis.tick_top()
    ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
    ax.tick_params(axis='y', labelsize=12)

    #for i in range(data.shape[0] + 1):
    #    ax.axhline(i - 0.5, color='ghostwhite', linewidth=0.5)
    #for j in range(data.shape[1] + 1):
    #    ax.axvline(j - 0.5, color='ghostwhite', linewidth=0.5)
    
    # colorbar
    cbar = plt.colorbar(im,ax=ax,ticks=[0.25,0.75],shrink=0.5,pad=0.02)
    cbar.ax.yaxis.set_tick_params(pad=11)  # increase pad from bar to labels
    cbar.ax.tick_params(size=0) 
    cbar.ax.set_yticklabels(['Normal=0','Dysfunctional≥1'],rotation=90)
    # Center the labels horizontally
    for label in cbar.ax.get_yticklabels():
        label.set_ha('center')  # horizontally center
        label.set_va('center')  # vertically center
    
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Predicted class',labelpad=10,fontsize=12)
    
    return 

##############################
def plotExp(ds,rcut,pvalcut):
    '''
    Plot experimental values. Raw and pvalue.
    '''
    
    residues = ds.variantNames
    sorted_residues = sorted(residues, key=lambda x: int(''.join(filter(str.isdigit,x))))

    ##########
    ## for electrophys labels
    
    ## color by label
    ## normal, lof, xx, nan, unc, gof, nm
    #colors = ['bisque','lightskyblue','k','k','darkgray','lightcoral','k']
    #colors = ['cornsilk','lightsalmon','k','k','darkgray','mediumorchid','k']
    colors = ['aliceblue','powderblue','k','k','goldenrod','dodgerblue','k'] # for fig 1B-H
    
    fig,ax=plt.subplots(figsize=(4,3)) # for manuscript
    ##fig,ax=plt.subplots(figsize=(8,6)) # for assessing twist variants 
    for name in sorted_residues:
        if ds.nanTrack[name]:
            continue
        val=ds.labelDictCont[name][0]
        pval=ds.labelDictCont[name][1]
        #plt.scatter(val,pval,color='bisque',edgecolor='k')        
        logpval=-math.log(pval,10) # volcano plots are usually -log10(pval)
        label=int(ds.labelDict[name])
        plt.scatter(val,logpval,color=colors[label],edgecolor='k') #,s=100) # size for unc plot 
        #ax.text(val,logpval,name,fontsize=12) # for unc plot 
            
    ax.set_xlabel("measurement value")
    ax.set_ylabel("-log10(p-value)")
    #ax.set_ylabel("p-value")
    if pvalcut is not None:
        #ax.axhline(y=-math.log(pvalcut[0],10), color='k', alpha=0.8, linestyle='--',lw=1)
        ax.axhline(y=-math.log(pvalcut[1],10), color='grey', alpha=0.3, linestyle='--',lw=1)        
        #ax.axhline(y=pvalcut[0], color='k', alpha=0.8, linestyle='--',lw=1)
        #ax.axhline(y=pvalcut[1], color='grey', alpha=0.8, linestyle='--',lw=1)
        
    [ax.axvline(x=i, color='k', alpha=0.3, linestyle='--',lw=1) for i in rcut]
    
    ax.tick_params(axis='both',labelsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(0.5)) # iks and tau deact
    #ax.set_xlim([-0.1,2.2]) # for tau deact
    
    #ax.xaxis.set_major_locator(MultipleLocator(10)) # v12
    #ax.xaxis.set_major_locator(MultipleLocator(1.0)) # tau act    
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # for tau_act
    
    #################################
    # make a lollipop plot instead
    figlol,axlol=plt.subplots(figsize=(3,8)) # for assessing twist variants
    #figlol,axlol=plt.subplots(figsize=(3,1.8)) # for assessing twist variants
    y=0
    yticks,ylabels=[],[]
    for name in sorted_residues:
        if ds.nanTrack[name]:
            # write ND
            axlol.hlines(y,xmin=0,xmax=0,color="grey",alpha=0.6)            
            axlol.text(0+0.1,y,'ND',va="center",fontsize=13,color='k')            
            yticks.append(y)
            ylabels.append(name)
            y+=1
            continue 
        val=ds.labelDictCont[name][0]
        pval=ds.labelDictCont[name][1]
        label=int(ds.labelDict[name])
        axlol.hlines(y,xmin=0,xmax=val,color="grey",alpha=0.6)
        axlol.scatter(val,y,s=50,edgecolor='k',zorder=3,color=colors[label])
        if pval<0.05:
            axlol.text(val+0.1,y,'*',va="center",fontsize=14,color='r')
            ## for v12:
            #if val>0:
            #    axlol.text(val+1.4,y,'*',va="center",fontsize=14,color='r')
            #else:
            #    axlol.text(val-3,y,'*',va="center",fontsize=14,color='r')
        #axlol.text(-0.1,y,name,ha="right",va="center")
        yticks.append(y)
        ylabels.append(name)
        y+=1
        
    axlol.set_xlim([0,2.6]) # iks , tau deact
    #axlol.set_xlim([-30,30]) # v12
    #axlol.set_xlim([0,4.01]) # tau act
    axlol.xaxis.set_major_locator(MultipleLocator(0.5)) #  for iks,tau deact
    #axlol.xaxis.set_major_locator(MultipleLocator(15)) # 15 for v12     
    #axlol.xaxis.set_major_locator(MultipleLocator(1)) # 1 for tau act
    axlol.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # for tau_act
    
    [axlol.axvline(x=i, color='k', alpha=0.3, linestyle='--',lw=1) for i in rcut]
    axlol.axvline(x=0, color='k', alpha=0.5, linestyle='-',lw=1)
    axlol.grid(True, which='both',alpha=0.1)    
    axlol.set_yticks(yticks)
    axlol.set_yticklabels(ylabels,fontsize=12)
    axlol.tick_params(axis='both',labelsize=12)
    axlol.invert_yaxis()
    figlol.tight_layout()
    figlol.savefig('output_exp_measures.pdf',format='pdf',bbox_inches="tight")
    #plt.show()
    
    ################################    
    ### distributions for measurement (no p-value), uncomment for  traficking labels
    #
    #vals = [
    #    ds.labelDictCont[name][0].item()
    #    for name in sorted_residues
    #    if not ds.nanTrack[name]
    #]
    #
    ## Create histogram with density=True for normalized distribution
    #figdist, axdist = plt.subplots(figsize=(4,3))
    #axdist.hist(vals, bins=15, density=True, color='bisque',edgecolor='k')
    #
    #[axdist.axvline(x=i, color='k', alpha=0.8, linestyle='--',lw=1) for i in rcut]
    #
    #axdist.tick_params(axis='both',labelsize=12)
    ##axdist.xaxis.set_major_locator(MultipleLocator(1)) # 0.5)) #1.0))    
    ##axdist.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # for tau_act
    #
    #xmin, xmax = axdist.get_xlim()
    #axdist.set_xlim(min(0, xmin), xmax)
    #
    #plt.tight_layout()
    #plt.show()
    #
    ######
    ### hz swarm plot
    #vararr = []
    #traf_colors = []
    #for i,name in enumerate(sorted_residues):
    #    if ds.nanTrack[name]:
    #        continue
    #    label=ds.labelDict[name]
    #    vararr.append(ds.labelDictCont[name][0].item())        
    #    traf_colors.append(colors[label])
    # 
    #y_cat = ["All"] * len(vararr)
    # 
    #figswarm, axswarm = plt.subplots(figsize=(4,1.5))
    #sbn.swarmplot(x=vararr, y=y_cat, hue=traf_colors,
    #              palette={"powderblue": "powderblue", "aliceblue": "aliceblue", "dodgerblue": "dodgerblue"},
    #              edgecolor='k',
    #              size=5,
    #              linewidth=0.6,
    #              ax=axswarm)
    # 
    #axswarm.set_ylabel("")
    #axswarm.set_xlim([-10,600])
    #
    #axswarm.tick_params(axis='both',labelsize=11)
    #axswarm.xaxis.set_major_locator(MultipleLocator(100)) # 0.5)) #1.0))
    #axswarm.xaxis.set_minor_locator(MultipleLocator(50)) # 0.5)) #1.0))        
    ##ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # for tau_act
    #axswarm.legend_.remove()  # if a legend exists
    #
    #[axswarm.axvline(x=i, color='k', alpha=0.3, linestyle='--',lw=1) for i in rcut]
    #
    #plt.tight_layout()
    #
    ######
    ### histogram 
    #fighist, axhist = plt.subplots(figsize=(4, 1.5))
    #palette = {
    #    "powderblue": "powderblue",
    #    "aliceblue": "aliceblue",
    #    "dodgerblue": "dodgerblue"
    #}
    #
    #vararr = np.array(vararr)
    #traf_colors = np.array(traf_colors)
    #
    ## align bins to cutoff vals
    #xmin, xmax = -10, 600
    #base_bins = np.linspace(xmin, xmax, 30)
    #bins = np.unique(np.concatenate([base_bins, rcut]))
    #bins = np.sort(bins)
    #print("N bins:",len(bins))
    #
    #data_by_color = [vararr[traf_colors == c] for c in palette.keys()]
    #axhist.hist(
    #    data_by_color,
    #    bins=bins,
    #    stacked=True,
    #    color=list(palette.values()),
    #    edgecolor='k',
    #    linewidth=0.5
    #)
    #
    ## cutoff lines
    #for x in rcut:
    #    axhist.axvline(x, color='k', alpha=0.3, linestyle='--', lw=1)
    #    
    #axhist.set_xlim(xmin, xmax)
    #axhist.set_ylabel("Count")
    #axhist.tick_params(axis='both', labelsize=10)
    #
    #axhist.yaxis.set_major_locator(MultipleLocator(10))
    #axhist.yaxis.set_minor_locator(MultipleLocator(5))
    #axhist.xaxis.set_major_locator(MultipleLocator(100))
    #axhist.xaxis.set_minor_locator(MultipleLocator(50))    
    #axhist.set_ylim([0,40])
    #                               
    #plt.tight_layout()
    #plt.show()

    return

##############################
def amValPlot(ds,rcut,pvalcut):
    '''
    Plot AM preds. 
    '''
    
    #minCut, maxCut = 0.3402, 0.5638 # see ../add_mpnn/model250331.mpnnn.py for cutoff det
    ## If i want to include ambiguous, uncomment below :
    minCut, maxCut = 0.5,0.5 

    residues = ds.variantNames
    sorted_residues = sorted(residues, key=lambda x: int(''.join(filter(str.isdigit,x))))

    am_raw_scores = np.stack([ds.featureDict[name].detach().cpu().numpy() for name in sorted_residues])[:,12]
    labels_np = (np.stack([ds.labelDict[name].detach().cpu().numpy() for name in sorted_residues]) >= 1).astype(int)

    # exclude AM ambiguous 
    am_label_np = np.full_like(am_raw_scores, fill_value=-1, dtype=int)
    am_label_np[am_raw_scores > maxCut] = 1
    am_label_np[am_raw_scores < minCut] = 0

    mask = am_label_np != -1
    filtered_probs = am_raw_scores[mask]    
    filtered_preds = am_label_np[mask]
    filtered_labels = labels_np[mask]
    
    mcc = skm.matthews_corrcoef(filtered_labels, filtered_preds)
    brier = skm.brier_score_loss(filtered_labels,filtered_probs)
    auprc = skm.average_precision_score(filtered_labels,filtered_preds)
    if len(set(filtered_labels)) > 1: 
        auroc = skm.roc_auc_score(filtered_labels,filtered_preds)
    else:
        auroc = -1
    print(f"AM MCC: {mcc:.3f} Brier: {brier:.3f}; AUROC: {auroc:.3f} ; AUPRC: {auprc:.3f} (excl. ambiguous)")

    fig,ax=plt.subplots(figsize=(3,7.5))
    for name in sorted_residues:
        am_val =ds.featureDict[name][12]
        ax.scatter(am_val,name,color='bisque',edgecolor='k')

    ax.invert_yaxis()
    ax.axvline(x=0.5638, color='k', alpha=0.3, linestyle='--',lw=1)
    ax.axvline(x=0.3402, color='k', alpha=0.3, linestyle='--',lw=1)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1)) # 0.5)) #1.0))
    ax.xaxis.set_major_locator(MultipleLocator(0.2)) # 0.5)) #1.0))    
    ax.grid(True, which='both',alpha=0.1)
    ax.tick_params(axis='both', which='both', labelsize=12)
    ax.set_xlim([-0.03,1.03])
    plt.tight_layout()
    plt.savefig('output_amvalplot.pdf',format='pdf',bbox_inches="tight")


    #figvar,axvar=plt.subplots(figsize=(20,3))
    #for name in sorted_residues:
    #    val=ds.labelDictCont[name][0]
    #    pval=ds.labelDictCont[name][1]
    #    am_val =ds.featureDict[name][12]
    #
    #    if am_val > maxCut:
    #        color='crimson'
    #        am_label = 1
    #    elif am_val < minCut:
    #        color='cornflowerblue'
    #        am_label = 0 
    #    else: # ambiguous
    #        color='bisque'
    #        am_label = -1
    #
    #    if ds.nanTrack[name]:
    #        val=-2
    #        pval=-2
    #        
    #    if am_label != (ds.labelDict[name]>=1).int() :
    #        ax.scatter(val,pval,color=color,edgecolor='k')        
    #        marker="*"
    #        color="green"
    #    else:
    #        marker="o"
    #
    #    axvar.scatter(name,val,color=color,marker=marker)
    #        
    #ax.set_xlabel("measurement value")
    ##ax.set_ylabel("-log10(p-value)")
    #ax.set_ylabel("p-value")
    #ax.set_title("only plotting wrong variants") 
    #if pvalcut is not None:
    #    #ax.axhline(y=-math.log(pvalcut,10), color='k', alpha=0.8, linestyle='--',lw=1)
    #    ax.axhline(y=pvalcut[0], color='k', alpha=0.8, linestyle='--',lw=1)
    #    ax.axhline(y=pvalcut[1], color='grey', alpha=0.8, linestyle='--',lw=1)         
    #[ax.axvline(x=i, color='k', alpha=0.8, linestyle='--',lw=1) for i in rcut]    
    #ax.set_yticks(ticks=np.arange(0,1.05,0.1))
    #
    #axvar.set_ylabel("measurement")
    #axvar.tick_params(axis='x',labelrotation=90)
    #[axvar.axhline(y=i, color='k', alpha=0.8, linestyle='--',lw=1) for i in rcut]        
    #
    #plt.tight_layout()

    return 

######################################################################
def plot_missing_variants(missing_cells, config, featuresCSVList, trainLabelsCSV,fname_model,cmap):
    ''' Plot variants in missing_cells cf to continuous label values. '''

    # for color 
    norm = plt.Normalize(vmin=0, vmax=1)  # needs to match the vmin/vmax in plotVarClass()
    
    # group missing vars by labelName
    grouped = {}
    for cell in missing_cells:
        labelName = cell["label"]
        grouped.setdefault(labelName, []).append(cell)
    
    # plot each labelName separately
    for labelName, cells in grouped.items():
        # Get correct label, rcut etc for this labelName
        label = next(label for label in config["labels"] if label["name"] == labelName)
        rcut, pvalcut, gof = label["rcut"], label["pvalcut"], label["gof"]

        # Load model for predictions
        savedModel = fname_model+labelName+".joblib" if fname_model is not None else ""
        idx0,idx1 = 0,15 # all features
        saved = joblib.load(savedModel)
        pipeline = saved['model'] 
        thresh = saved['thresh']  

        ##############################
        ## plot like log(pval) vs measurement
        fig,ax=plt.subplots(figsize=(4,3)) # 4,3 for with pval
        for cell in cells:
            variant = cell["variant"]
            # get variant data            
            dsvar = VariantDataset(featuresCSVList,trainLabelsCSV,labelName,rcut,pvalcut,gof,True,False,False,False,False,None,variant)
            val=dsvar.labelDictCont[variant][0]
            pval=dsvar.labelDictCont[variant][1]
            if (pval.item() < 0 ) : # no value for trafficking vars
                pval=1.0
                print("WARNING!!!!! Plots will be generaated with measurement+pval. DO NOT use plots with trafficking measurements.")
            logpval=-math.log(pval,10) # volcano plots are usually -log10(pval)
            # get variant prediction
            features_np = dsvar.featureDict[variant].detach().cpu().numpy()    
            output = pipeline.predict_proba(features_np[idx0:idx1].reshape(1,-1))
            pred = int(output[0,1]>thresh)
            # plot 
            ax.scatter(val,logpval,color=cmap(norm(pred)),edgecolor='k',linewidth=2,s=100)
            #y_jitter = 1 + np.random.normal(0, 0.02)  # std dev controls spread
            #ax.scatter(val,y_jitter,color=cmap(norm(pred)),edgecolor='k')
            #ax.text(val,logpval,variant,fontsize=10)
                
        ax.set_title(labelName)
        ax.set_xlabel("measurement value")
        ax.set_ylabel("-log10(p-value)")
        if pvalcut is not None:
            ax.axhline(y=-math.log(pvalcut[1],10), color='grey', alpha=0.3, linestyle='--',lw=1)
        [ax.axvline(x=i, color='k', alpha=0.3, linestyle='--',lw=1) for i in rcut]
        ax.tick_params(axis='both',labelsize=12)
        
        ax.xaxis.set_major_locator(MultipleLocator(0.1)) # 0.5)) #1.0))
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # for tau_act
        # to extend the unc plots a bit more to be closer to fig 1 plot        
        #ax.set_xlim([0,2.6]) # iks
        #ax.set_xlim([-11,11]) # v12
        #ax.set_xlim([0,2.2]) # tau_act, deact
        
        #ax.set_ylim([0.9,1.1])
        ax.set_ylim([-0.09,1.7])

        # tau_act zoom        
        ax.set_xlim([0.25,0.8])
        ax.set_ylim([-0.09,0.51]) 
        
    return 

######################################################################
def plotCategories(labels,testpreds):
    ''' Establish 4 categories. Analyze predictions, var distributions of these categories.'''
    
    categories = defaultdict(set)

    variants = set(key.rsplit('_', 1)[1] for key in testpreds.keys())
    rows = sorted(variants,key=getpos)        
    cols = labels    

    data = np.full((len(rows),len(cols)),np.nan) # initialize
    for i , var in enumerate(rows): # save data
        for j,label in enumerate(cols):
            key = f"{label}_{var}"
            pred = testpreds[key]["preds"]
            data[i,j] = pred
    
    for i, var in enumerate(rows): 
        predrow = data[i,:]
        if predrow[0] == 1: # iks dysfunctional
            categories["cat1"].add(var)
        if np.any(predrow[1:4] == 1): # v12 or time const dysfunctional
            categories["cat2"].add(var)
        if np.any(predrow[5:] == 1):  # mistrafficking
            categories["cat3"].add(var)
        if np.all(predrow == 0):     # normal 
            categories["cat4"].add(var)

    # venn diagram, dysfunctional categories
    cmap = plt.cm.viridis  # any matplotlib colormap
    colors = [cmap(x) for x in np.linspace(0, 1, 4)]  # 4 colors for 4 circles

    plt.figure(figsize=(6,6))
    v = venn3(
        [categories["cat1"], categories["cat2"], categories["cat3"]],
        set_labels=("","",""), #iks dysfunctional", "v12, time constants dysfunctional", "mistrafficker")
        set_colors=colors[0:3]
    )
    for subset_id in ['100', '010', '001', '110', '101', '011', '111']:
        patch = v.get_patch_by_id(subset_id)
        if patch:  # some overlaps may be empty
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
    
    # add separate circle for normal category (mutually exclusive)
    cat4_size = len(categories['cat4'])
    ref_size = max(len(categories['cat1']), len(categories['cat2']), len(categories['cat3']), 1)
    # radius proportional to sqrt(count) for area-based visual
    radius = np.sqrt(cat4_size / ref_size) * 0.5  # 0.5 is a scaling factor for visual fit
    
    # position: place to the right and slightly up
    x_pos = 1.6
    y_pos = 0.6
    
    # add circle
    circle = plt.Circle((x_pos, y_pos), radius, facecolor=colors[3], alpha=0.3, edgecolor='k',linewidth=2) #, label=f'Normal ({cat4_size})')
    plt.gca().add_patch(circle)

    # create legend
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.6, label=f'Iks dysfunctional ({len(categories["cat1"])})'),
        Patch(facecolor=colors[1], alpha=0.6, label=f'v12, time constants dysfunctional ({len(categories["cat2"])})'),
        Patch(facecolor=colors[2], alpha=0.6, label=f'mistrafficker ({len(categories["cat3"])})'),
        Patch(facecolor=colors[3], alpha=0.3, label=f'Normal ({cat4_size})')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Venn Diagram: 3 overlapping categories + size-represented 4th category")
    plt.axis('equal')
    
    plt.show()
    
    return 

######################################################################
def generate_shap_waterfall_custom(labelname,label_thresh,residue,explanation,max_display=10):
    # custom implementation of shap.waterfall_plot()
    
    values = explanation.values
    base_value = explanation.base_values
    feature_names = explanation.feature_names
    
    # sort by absolute importance
    order = np.argsort(-np.abs(values))
    values = values[order]
    feature_names = np.array(feature_names)[order]
    
    # limit number of features
    values = values[:max_display]
    feature_names = feature_names[:max_display]

    # cumulative contributions
    cumulative = base_value + np.cumsum(values)

    fig, ax = plt.subplots(figsize=(9, 4))

    prev = base_value

    spacing = 0.4  # < 1 = tighter, > 1 = looser
    for i, (val, name) in enumerate(zip(values, feature_names)):
        color = "lightblue" if val > 0 else "seashell"
        y = i * spacing        

        ax.barh(
            y,
            val,
            left=prev,
            color=color,
            edgecolor="black",
            height=0.3
        )

        ## optional text label
        #ax.text(
        #    prev + val/2,
        #    y,
        #    f"{val:.2f}",
        #    va="center",
        #    ha="center",
        #    fontsize=9
        #)
        
        prev += val

    final_value = base_value + values.sum()

    # reference lines
    ax.axvline(base_value, linestyle="solid", color="k",
               label=f"Mean model probability",alpha=0.4,linewidth=2)
    # threshold line
    ax.axvline(label_thresh, linestyle="solid", color="darkorange",
               label=f"Dysfunctional threshold",alpha=0.9,linewidth=2)
    pred_label= "Dysfunctional" if final_value>label_thresh else "Normal"
    ax.axvline(final_value, linestyle="dashed", color="mediumorchid",
               label=f"Model prediction: {pred_label}",alpha=1,linewidth=2)

    ax.set_yticks(np.arange(len(values)) * spacing)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()

    #ax.set_xlabel("Model output",fontsize=12)
    ax.set_xlim([0,1])
    ax.legend(fontsize=11,facecolor='white',framealpha=0.8)
    ax.tick_params(axis='both',labelsize=12)

    #ax.set_title(f"Waterfall plot - variant {residue}")
    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    ax.grid(True,alpha=0.1)
    ax.set_axisbelow(True)
    plt.savefig(f"shap_waterfall_{labelname}_{residue}.pdf",dpi=300)
    plt.close()

    return 

######################################################################
def generate_feat_by_class_violin_plot(all_labels,featclass,ofname): 
    ''' 
    Code returns a violin plot comparing structural descriptors b/en normal/dysf variants.
    '''

    featuresList_biophys=['change no. H acceptor sites','change no. H donor sites','change volume of aa','mutat AA hydrophobicity','mutant AA polarizability','functional density (polarizability 6.5 A)','functional density (polarizability 12 A)','functional density (hydrophobicity 1 A)','functional density (hydrophobicity 6.5 A)','distance from channel pore axis','burial in membrane']

    fig,axes = plt.subplots(len(featuresList_biophys),1,figsize=(7,14))
    colors = ['aliceblue','steelblue'] # gt labels here, so use same benign color as fig 1, a blue in b/en powdrblue and dodgerblue
    colors = [(*mcolors.to_rgba(c)[:3], 0.4) for c in colors]

    ### sanity checking violin plots with no visible iqr box 
    ##key = 'change no. H acceptor sites_iks'
    #key = 'mutat AA hydrophobicity_iks'
    #b = featclass[key]['benign_feat']
    #p = featclass[key]['dysfunctional_feat']
    #
    #print(f"benign     — median: {np.median(b):.4f}, IQR: {np.percentile(b,75)-np.percentile(b,25):.4f}, % zeros: {(b==0).mean()*100:.1f}%")
    #print(f"dysfunctional — median: {np.median(p):.4f}, IQR: {np.percentile(p,75)-np.percentile(p,25):.4f}, % zeros: {(p==0).mean()*100:.1f}%")
    
    for i, ifeat in enumerate(featuresList_biophys):
        ax = axes[i]
        feats, labels,class_list= [],[],[]
        for labelname in all_labels:
            key = ifeat + "_" + labelname
            if key not in featclass:
                continue 
            for v in featclass[key]["benign_feat"]:
                feats.append(v)
                labels.append(labelname)
                class_list.append("benign")
            for v in featclass[key]["dysfunctional_feat"]:
                feats.append(v)
                labels.append(labelname)
                class_list.append("dysfunctional")

        # sanity check 
        for labelname in all_labels:
            count = sum(1 for l, c in zip(labels, class_list) if l == labelname and c == "benign")
            print(f"N counts for {ifeat} and {labelname} and benign: {count}" )
            count = sum(1 for l, c in zip(labels, class_list) if l == labelname and c == "dysfunctional")        
            print(f"N counts for {ifeat} and {labelname} and dysfunctional: {count}" )         

            # violin plot misleading for discrete variable            
            if ifeat not in ["change no. H acceptor sites","change no. H donor sites"]:
                sbn.violinplot(
                    ax=ax,            
                    x=labels,
                    y=feats,
                    hue=class_list,
                    inner=None, #"box",
                    palette=colors,
                    cut=0,
                    scale="width",
                    hue_order=["benign","dysfunctional"],
                    split=True,
                )
                sbn.boxplot(
                    ax=ax,            
                    x=labels,
                    y=feats,
                    hue=class_list,
                    width=0.2,
                    linewidth=0.5,
                    #gap=0.2,
                    palette=colors,
                    hue_order=["benign","dysfunctional"],
                    flierprops=dict(marker='o',markersize=2,markerfacecolor='white',markeredgecolor='k',markeredgewidth=0.5),
                    boxprops={'zorder':2},
                    medianprops={'color':'red','linewidth':0.4}                    
                    #legend=False
                )
                # recolor dysfunctional box lines 
                # seaborn draws boxes in hue order, so every 2nd box belongs to dysfunctional
                for j, artist in enumerate(ax.artists):
                    if j % 2 == 1:  # dysfunctional is index 1 in hue_order
                        artist.set_edgecolor('azure')
                        #artist.set_linecolor('gainsboro')
                        for k in range(6): # 6 lines per box
                            if k == 4: # don't change median line color
                                continue 
                            line = ax.lines[j*6+k]
                            line.set_color('azure')
                            

            else: # box plot only for change no. H * sites, need to adjust fill colors
                sbn.boxplot(
                    ax=ax,            
                    x=labels,
                    y=feats,
                    hue=class_list,
                    width=0.2,
                    linewidth=0.5,
                    #gap=0.2,
                    palette=colors,
                    hue_order=["benign","dysfunctional"],
                    flierprops=dict(marker='o',markersize=2,markerfacecolor='white',markeredgecolor='k',markeredgewidth=0.5),
                    medianprops={'color':'red','linewidth':0.4}
                    #boxprops={'zorder':2}
                    #legend=False
                )
                            
        ax.set_xlabel("")
        ax.set_ylabel(ifeat)
        ax.grid(True,alpha=0.3)    
        ax.tick_params(axis='both',labelsize=9)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))

        ## only keep legend on first subplot
        if i==0:
            ax.legend(fontsize=9)
        else:
            ax.get_legend().remove()

        #ax.yaxis.label.set_visible(False) # write out labels in inkscape
        if i < len(featuresList_biophys) - 1:
            ax.tick_params(axis='x',labelbottom=False)

    # manual formatting tweaks
    axes[7].yaxis.set_major_locator(MultipleLocator(0.3))    
    axes[7].set_ylim([-0.34,0.6])

    # for discrete features
    axes[0].yaxis.set_major_locator(MultipleLocator(1))
    axes[1].yaxis.set_major_locator(MultipleLocator(2))
    axes[1].yaxis.set_minor_locator(MultipleLocator(1))            
    axes[1].set_ylim([-3.2,3.2])
    #plt.tight_layout()
    
    plt.savefig(ofname, bbox_inches="tight",dpi=300)  
    plt.show()
    
    return
