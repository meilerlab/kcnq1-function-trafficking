import torch
from torch.utils.data import Dataset

class VariantDataset(Dataset):
    """
    Custom Dataset for loading variant features and labels.
    """
    def __init__(self, featurecsv_list, labelcsv, labelname,rcut,pvalcut,gof,excludeWT,exclNM,exclUNC,VSDonly,PDonly,excludeVars,includeVars):
        self.featureDict = {}
        self.labelDict = {}
        self.nanTrack = {}
        self.labelDictCont = {}
        # Read features 
        self.readFeatures(featurecsv_list)
        # Read and process labels
        self.readLabels(labelcsv,labelname, rcut,pvalcut,gof,exclNM,exclUNC)
        self.readLabelsCont(labelcsv,labelname,pvalcut) # pvalcut is just to specify if pval is an input

        # Keep only variants present in both features and labels
        common_variants = set(self.featureDict.keys()) & set(self.labelDict.keys())
        if (VSDonly):
            filtered_variants = {var for var in common_variants if int(''.join(filter(str.isdigit, var))) <= 249}
            common_variants = filtered_variants
        if (PDonly):
            filtered_variants = {var for var in common_variants if int(''.join(filter(str.isdigit, var))) > 257}
            common_variants = filtered_variants

        self.featureDict = {k: self.featureDict[k] for k in common_variants}
        self.labelDict = {k: self.labelDict[k] for k in common_variants}

        if excludeWT:     # Exclude wild-type variants (where first and last character are the same)
            self.variantNames = [name for name in self.featureDict.keys() if name[0] != name[-1]]
        else:
            self.variantNames = list(self.featureDict.keys())

        #if excludeVars is not None: # exclude some variants from dataset
        self.variantNames = [name for name in self.variantNames if name not in excludeVars] if excludeVars is not None else list(self.variantNames)
        #if includeVars is not None: # include only specified variants froom dataset
        self.variantNames = [name for name in self.variantNames if name in includeVars] if includeVars is not None else list(self.variantNames)        
            
        # Ensure feature and label dictionaries have the same order
        self.variantNames.sort()
        self.featureDict = {k: self.featureDict[k] for k in self.variantNames}
        self.labelDict = {k: self.labelDict[k] for k in self.variantNames}
        self.labelDictCont = {k: self.labelDictCont[k] for k in self.variantNames}

    def __len__(self):
        return len(self.variantNames)

    def __getitem__(self, idx):
        variant_name = self.variantNames[idx]
        return self.featureDict[variant_name], self.labelDict[variant_name] #,self.amDict[variant_name]
    
    def readFeatures(self, featurecsv_list):
        """
        Reads features from a list of CSV files and stores them in a dictionary.
        """
        for featurecsv in featurecsv_list:
            with open(featurecsv, 'r') as f:
                for line in f:
                    parts = line.strip().split(",")
                    name = parts[0].strip()
                    sel_features = parts[1:]
                    features = torch.tensor([float(x) for x in sel_features])
                    if name in self.featureDict:
                        # Concatenate features
                        self.featureDict[name] = torch.cat((self.featureDict[name], features))
                    else:
                        self.featureDict[name] = features
                    
    def readLabels(self, labelcsv,labelname,rcut,pvalcut,gof,exclNM,exclUNC):
        """
        Reads labels from a CSV file, processes them according to rcut, and stores them in a dictionary.
        """
        with open(labelcsv, 'r') as f:
            header = f.readline().strip().split(",")
            colidx =  {col: i for i, col in enumerate(header)}
            for line in f:
                parts = line.strip().split(",")
                variant = parts[colidx['variant']].strip()
                val_str = parts[colidx[labelname]].strip()
                #val = float(val_str) if (val_str != "nan" or val_str != "not_measured") else 0.0
                val = 0.0 if (val_str == "nan" or val_str == "not_measured") else float(val_str)
                isnan = val_str == "nan"
                is_not_measured = val_str == "not_measured"

                label = None
                if isnan:
                    label = 3
                elif is_not_measured:
                    label = 6
                    if exclNM:
                        continue 
                #elif is_iks_lt_17:
                #    label = 1
                elif val < rcut[0]  or val > rcut[1]:
                    if pvalcut is not None:
                        pval_col = labelname + "_pval"
                        pval_str = parts[colidx[pval_col]].strip()
                        assert pval_str != "nan","label without a value has a non-nan p-value"
                        assert pval_str != "not measured","label without a value has a non-nan p-value"
                        pval = float(pval_str)
                        if pval <= pvalcut[0]: # dysfunctional
                            label=setDysfunc(val,rcut,gof)
                        elif pval <= pvalcut[1]: # likely dysfunctional 
                            #label = 2
                            label=setDysfunc(val,rcut,gof) 
                        else:
                            # exclude variants that may be dysfunction but p value is not statistically significant 
                            label = 4 # to identify
                            #print("NOT EXCLUDING UNCERTAIN VARIANTS")
                            if exclUNC:
                                continue 
                    else:
                        label=setDysfunc(val,rcut,gof)                        
                else:
                    label = 0 # normal function

                assert label is not None, "A variant label is not appropriately assigned."
                
                self.labelDict[variant] = torch.tensor(label)
                self.nanTrack[variant] = torch.tensor(isnan)  # Store NaN tracking for this entry

    def readLabelsCont(self, labelcsv,labelname,pvalcut): # read raw measurement value 
        """
        Reads measurement values (labels) from a CSV file and stores them in a dictionary.
        """
        with open(labelcsv, 'r') as f:
            header = f.readline().strip().split(",")
            colidx =  {col: i for i, col in enumerate(header)}
            for line in f:
                parts = line.strip().split(",")
                variant = parts[colidx['variant']].strip()

                val_str = parts[colidx[labelname]].strip()
                val = 0.0 if (val_str == "nan" or val_str == "not_measured") else float(val_str)                
                #val = float(val_str) if val_str != "nan" else 0
                if pvalcut is not None:
                    pval_col = labelname + "_pval"
                    pval_str = parts[colidx[pval_col]].strip()
                    pval = 1 if ( pval_str == "nan" or val_str == "not_measured") else float(pval_str) 
                else:
                    pval=-1 # dummy placeholder 
                labels = torch.zeros(2)
                labels[0]=val
                labels[1]=pval
                self.labelDictCont[variant] = labels

##############################
class TestDataset(Dataset):
    """
    Custom Dataset for loading variant features for test variants (no labels). 
    """
    def __init__(self, featurecsv_list, labelname,excludeWT,VSDonly,PDonly,excludeVars,includeVars):
        self.featureDict = {}
        # Read features 
        self.readFeatures(featurecsv_list)

        common_variants = set(self.featureDict.keys())

        if (VSDonly):
            filtered_variants = {var for var in common_variants if int(''.join(filter(str.isdigit, var))) <= 249}
            common_variants = filtered_variants
        if (PDonly):
            filtered_variants = {var for var in common_variants if int(''.join(filter(str.isdigit, var))) > 257}
            common_variants = filtered_variants

        self.featureDict = {k: self.featureDict[k] for k in common_variants}

        if excludeWT:     # Exclude wild-type variants (where first and last character are the same)
            self.variantNames = [name for name in self.featureDict.keys() if name[0] != name[-1]]
        else:
            self.variantNames = list(self.featureDict.keys())

        #if excludeVars is not None: # exclude some variants from dataset
        self.variantNames = [name for name in self.variantNames if name not in excludeVars] if excludeVars is not None else list(self.variantNames)
        #if includeVars is not None: # include only specified variants froom dataset
        self.variantNames = [name for name in self.variantNames if name in includeVars] if includeVars is not None else list(self.variantNames)        
            
        # Order featureDict
        self.variantNames.sort()
        self.featureDict = {k: self.featureDict[k] for k in self.variantNames}

    def __len__(self):
        return len(self.variantNames)

    def __getitem__(self, idx):
        variant_name = self.variantNames[idx]
        return self.featureDict[variant_name]
    
    def readFeatures(self, featurecsv_list):
        """
        Reads features from a list of CSV files and stores them in a dictionary.
        """
        for featurecsv in featurecsv_list:
            with open(featurecsv, 'r') as f:
                for line in f:
                    parts = line.strip().split(",")
                    name = parts[0].strip()
                    sel_features = parts[1:]
                    features = torch.tensor([float(x) for x in sel_features])
                    if name in self.featureDict:
                        # Concatenate features
                        self.featureDict[name] = torch.cat((self.featureDict[name], features))
                    else:
                        self.featureDict[name] = features
                
##############################
def setDysfunc(val,rcut,gof):

    label = 0 
    
    if gof=="larger": #default 
        if val < rcut[0]: # lof 
            label = 1
        elif val > rcut[1]: # gof
            label = 5
    elif gof=="smaller": 
        if val > rcut[0]: # lof
            label = 1
        elif val < rcut[1]: # gof
            label = 5
    else:
        print("unknown gof setting")
        assert(1)

    return label 
