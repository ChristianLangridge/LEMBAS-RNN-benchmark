# LEMBAS-RNN benchmarking project

A benchmarking framework for evaluating a novel biologically-informed RNN architecture [LEMBAS-RNN](https://github.com/czhang0701/DLBionet) designed by Li *et al*, 2025 [UNPUBLISHED] against standard regression baseline models to predict target gene expression from transcription factor (TF) expression values in human liver tissue (bulk-RNA-seq)

**Rotational project in Roger Williams Institute for Liver Studies at King's College London with Dr. Cheng Zhang**


## Table of Contents

[Jump to Background] ### - **Background**
[Jump to Architecture Overview] ### - **Architecture Overview**
[Jump to Models benchmarked] ### - **Models Benchmarked**
[Jump to Results summary] ### **Results Summary**
[Jump to Repository structure] ### **Repository Structure**
[Jump to Installation] ### **Installation**
[Jump to Data Requirements] ### **Data Requirements**
[Jump to Usage] ### **Usage**
[Jump to SHAP Explainability] ### **SHAP Explainability**
[Jump to Known Issues & Limitations] ### **Known Issues & Limitations**
[Jump to Contributing] ### **Contributing** 
[Jump to License] ### **License**


## Background

Gene expression prediction is a central interest in Systems Biology. Transcription factors (TFs) regulate the expression of target genes through a complex biological signalling network. Classical machine learning approaches (linear regression, tree-based models) treat this task as a black-box tabular problem and do not take into account for previously validated network topology. 

LEMBAS-RNN builds on this strategy: using a central RNN constrained by a real biological signalling network represented in latent space `network.tsv`, embedding known TF-target regulatory interactions directly for model training using Michaelis-Menten-like (MML) activtion functions. This produces a model that is both biologically-informed and compatible with post-hoc explainability techniques like [SHAP analysis](https://github.com/shap/shap). 

This repository benchmarks LEMBAS-RNN against two standard baseline models - Multiple Linear Regression (MLR) and XGBoost Random Forest Regression (XGBRF) - using a held-out test set and an independent external validation cohort of human liver bulk-RNA-seq data provided by [Yang H. *et al*, 2025](https://pubmed.ncbi.nlm.nih.gov/39889710/)


## Architecture Overview

LEMBAS-RNN is made of separate modules:

<p align="center"> <img width="601" height="195" alt="image" src="https://github.com/user-attachments/assets/6feb6fc4-92d3-4d0f-9f3e-849a1f93eaba" />


**Key Designs**

*Add here if desired*


## Models Benchmarked

| **Model** | **Type** | **Hyperparameter choices** |
| --------- | -------- | -------------------------- |
| MLR | Multiple Linear Regressor | n_jobs=-1, sklearn LinearRegression() |
| XGBRF | XGBoost Random Forest Regressor | n_estimators=3, objective=reg:squarederror,random_state=888, trained in batches of 1,000 targets | 
| LEMBAS-RNN | Biologically-informed RNN | target_steps=150, max_steps=10, exp_factor=50, leak=0.01, tolerance=1e-20 |


## Results Summary 

All metrics cimputed on unseen validation set (262 samples, 16,100 target genes)

Aggregate Performance (Validation Set)

| **Model** | **Flattened R²** | **Pearson's R** | **RMSE** | **MAE** |
| --------- | ---------------- | ------------------------- | -------- | ------- |
| MLR | 0.9528 | 0.9764 | 0.1261 | 0.0726 |
| XGBRF | 0.9346 | 0.9669 | 0.1484 | 0.0884 |
| LEMBAS-RNN | 0.8441 | 0.9246 | 0.2290 | 0.1576 |

> **Note on R² methods:** `sklearn`'s `.score()` computes uniform-average R² across genes. The `compute_metrics()` function in this repo computes variance-weighted (flattened) R², which is substantially higher because the model model performance is heterogenous and disproportionately better on medium-variance genes.


## Repository Structure

EMBAS-RNN-benchmark/
│
├── benchmarking/
│   ├── data preprocessing/
│   │   └── model_boilerplate_remote.py     # Shared data loading & train/test split
│   │
│   ├── model scripts/
│   │   ├── LEMBAS-RNN/
│   │   │   ├── RNN_reconstructor.py        # Core model classes (SignalingModel, BioNet,
│   │   │   │                               #   ProjectInput, ProjectOutput)
│   │   │   └── RNN_testing.ipynb           # Model loading, inference & diagnostics
│   │   ├── MLR/
│   │   │   └── MLR.ipynb                   # MLR training, evaluation & k-fold CV
│   │   └── XGBRF/
│   │       └── XGBRF.ipynb                 # Batched XGBRF training & evaluation
│   │
│   └── figures/
│       ├── Model-fitting/
│       │   └── Fig1(fitting).ipynb         # Observed vs predicted (training set)
│       ├── Model-testing/
│       │   └── Fig1.ipynb                  # Observed vs predicted (test set)
│       ├── Model-validation/
│       │   └── Fig1(validation).ipynb      # External validation cohort figures
│       └── SHAP integration/
│           └── Fig1(SHAP).ipynb            # SHAP waterfall plots across all models
│
├── config/
│   ├── predictions/
│   │   ├── model_load.py                   # Loads all three trained models + metrics helpers
│   │   └── model_train_test_predictions.py # Generates predictions for all models
│   ├── figures/
│   │   └── figure1_agg_pearson_R.py        # Figure generation & publication styling
│   └── SHAP/
│       ├── SHAP_generation_baseline.py     # SHAP for MLR (Linear) & XGBRF (Tree)
│       ├── SHAP_generation_RNN.py          # SHAP for RNN (GradientExplainer)
│       └── SHAP_value_test_load_plot.py    # SHAP loading, validation & poster plotting
│
├── dep/                                    # External dependencies / submodules
├── .vscode/                                # Editor settings
├── .gitignore
├── LICENSE                                 # BSD-3-Clause
└── README.md
