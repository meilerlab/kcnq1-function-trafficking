# KCNQ1 variant effect function and trafficking 

## Classification models for KCNQ1 variants distinguish functional and trafficking effects to enhance pathogenicity interpretation

### Overview
This repository contains code for predicting KCNQ1 potassium ion channel protein fitness metrics using random forest classifiers. The models are trained on electrophysiological and trafficking data. The models predict how KCNQ1 variants affect channel function or trafficking, improving clinical interpretation of genetic variants.

### Associated Preprint
Chang-Gonzalez AC, Bell EW, Vanoye CG, Guadarrama E, Desai RR, DeKeyser J-M, Butcher KR, Scott T, Sanders CR, George AL Jr., Ledwitch KV, Meiler J. Classification models distinguish functional and trafficking effects of KCNQ1 variants to enhance variant interpretation. bioRxiv 2025. doi: 10.1101/2025.10.31.685955

### The easiest way to use this repository is to access pre-computed predictions from **`kcnq1_predictions.csv`**. 

### Repository Structure
```
.
├── main.py                          # Main script for training, evaluating, running classification models 
├── variant_dataset.py               # Variant data processing
├── plot_scripts.py                  # Visualization functions
├── environment.yml                  # Conda environment 
├── input_8sik_clean.pdb 	     # Input PDB
├── features_8sik_clean.w_am_mpnn_esm.csv  # Pre-generated features
├── kcnq1_all_measurements.csv       # Wet-lab experimental measurements
├── kcnq1_predictions.{csv,hdf5}     # Pre-computed predictions for KCNQ1 variants
├── saved_models/                    # Trained models and hyperparameters
├── sample_input_yaml/               # Example input configurations
├── feature_generation/              # Files for generating model features
├── list_variant_subsets/            # Lists of variant subsets
├── additional_analyses_scripts/     # Standalone analysis scripts
└── s4_data/			     # Electrophysiology data for S4 variants
```

### Setup
This project is built using Python. We recommend using a conda environment to manage dependencies.
```
conda env create -f environment.yml
conda activate kcnq1pred
```

### Usage
The main pipeline is executed via:
```
python main.py <input_config.yaml> 
```
The specifications in the YAML determine how the program will run. See sample YAML files in `sample_input_yaml/`. 
- ``crossval_q1TrainVal_procedures.yaml`` : uses all variants in `kcnq1_all_measurements.csv` with features defined in `features_8sik_clean.w_am_mpnn_esm.csv` to train random forest classifiers, saves cross-validation metrics, and plots. It also saves best model and hyperparameters for later use.
- ``train_test_excluding_set.yaml`` : excludes a select subset of variants for training, uses it for testing
- ``test_subset.yaml`` : uses saved model to get predictions for variant subset 

### Notes
- The "Ipeak" label in the manuscript text is "Iks" in project files and YAML
- The first 12 features were generated using the BCL (https://github.com/BCLCommons/bcl.git)
<details>
<summary>Column headers for <code>features_8sik_clean.w_am_mpnn_esm.csv</code></summary>
Δ Num. H acceptor sites, 
Δ Num. H donor sites, 
Δ Volume AA, 
Δ PSSM NR, 
Mutant AA hydrophobicity, 
Mutant AA polarizability, 
Funct. density (polarizability 6.5 Å), 
Funct. density (polarizability 12 Å), 
Funct. density (hydrophobicity 1 Å), 
Funct. density (hydrophobicity 6.5 Å), 
Distance from channel pore axis, 
Burial on membrane, 
AM, 
ProteinMPNN, 
ESM
</details>

### Citation
If you use this code, the models, datasets, or preditions in your research, please cite the corresponding preprint https://doi.org/10.1101/2025.10.31.685955. 
