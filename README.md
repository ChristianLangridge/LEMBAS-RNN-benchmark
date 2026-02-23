# LEMBAS-RNN benchmarking project
[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=flat-square)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Data Requests](https://img.shields.io/badge/data_requests-zcbtcl9%40ucl.ac.uk-informational?style=flat-square&logo=gmail)](https://github.com/ChristianLangridge)

A benchmarking framework for evaluating a novel biologically-informed RNN architecture [LEMBAS-RNN](https://github.com/czhang0701/DLBionet) designed by Li *et al*, 2025 [UNPUBLISHED] against standard regression baseline models to predict target gene expression from transcription factor (TF) expression values in human liver tissue (bulk-RNA-seq).

***Rotational project in Roger Williams Institute for Liver Studies at King's College London with Dr. Cheng Zhang***

---

# Table of Contents

- [Background](#background)
- [Architecture Overview](#architecture-overview)
- [Models Benchmarked](#models-benchmarked)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [SHAP Interpretability](#shap-interpretability)
- [Known Issues & Limitations](#known-issues--limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Background

Gene expression prediction is a central interest in Systems Biology. Transcription factors (TFs) regulate the expression of target genes through a complex biological signalling network. Classical machine learning approaches (linear regression, tree-based models) treat this task as a black-box tabular problem and do not take into account for previously validated network topology. 

LEMBAS-RNN builds on this strategy: using a central RNN constrained by a real biological signalling network represented in latent space `network.tsv`, embedding known TF-target regulatory interactions directly for model training using Michaelis-Menten-like (MML) activation functions. This produces a model that is both biologically-informed and compatible with post-hoc explainability techniques like [SHAP analysis](https://github.com/shap/shap). 

This repository benchmarks LEMBAS-RNN against two standard baseline models - Multiple Linear Regression (MLR) and XGBoost Random Forest Regression (XGBRF) - using a held-out test set and an independent external validation cohort of human liver bulk-RNA-seq data provided by [Yang H. *et al*, 2025](https://pubmed.ncbi.nlm.nih.gov/39889710/).

---

## Architecture Overview

LEMBAS-RNN is made of separate modules:

<p align="center"> <img width="601" height="195" alt="image" src="https://github.com/user-attachments/assets/6feb6fc4-92d3-4d0f-9f3e-849a1f93eaba" />


**Key Designs**

**Projection input unit** - scales `network.tsv` with expression data to create a biologically-informed graph input in latent space for DL training  

**Bionet signaling network** - the RNN engine iteratively updates node states in the graph, converging over time to a final output with respect to training parameters

**Projection output layer** - extracts nnode states and applies linear transformation to predict target gene expression values as a final model output

---

## Models Benchmarked

| **Model** | **Type** | **Hyperparameter choices** |
| --------- | -------- | -------------------------- |
| MLR | Multiple Linear Regressor | n_jobs=-1, sklearn LinearRegression() |
| XGBRF | XGBoost Random Forest Regressor | n_estimators=3, objective=reg:squarederror,random_state=888, trained in batches of 1,000 targets, xgb XGBRFRegressor() | 
| **LEMBAS-RNN** | **Biologically-informed RNN** | **target_steps=150, max_steps=10, exp_factor=50, leak=0.01, tolerance=1e-20** |

---

## Results Summary 

All metrics computed on unseen validation set (262 samples, 16,100 target genes)

<ins>Aggregate Performance (Validation Set)</ins>

| **Model** | **Flattened R²** | **Pearson's R** | **RMSE** | **MAE** |
| --------- | ---------------- | ------------------------- | -------- | ------- |
| MLR | 0.9528 | 0.9764 | 0.1261 | 0.0726 |
| XGBRF | 0.9346 | 0.9669 | 0.1484 | 0.0884 |
| **LEMBAS-RNN** | **0.8441** | **0.9246** | **0.2290** | **0.1576** |

> **Note on R² methods:** `sklearn`'s `.score()` computes uniform-average R² across genes. The `compute_metrics()` function in this repo computes variance-weighted (flattened) R², which is substantially higher because the model model performance is heterogenous and disproportionately better on medium-variance genes.

---

## Repository Structure

### config

All scripts related to configuration of model predictions, figure generation, latency testing and SHAP integration on local machine.

### dep

Includes .yml file with all dependencies used on development machine. 

### doc 

Includes all documentation relevant to the design of LEMBAS-RNN (Li *et al*, 2025) and the benchmarking framework (my rotational report, 2026).

### run

All scripts related to the running of the benchmarking framework for data preprocessing, model training/initialisation and figure generation.

### test

Includes all unit/integration testing so far. More are planned.

---

## Installation 

**Prerequisites**

- Python 3.12+
- A CUDA-capable GPU is recommended for RNN training (CPU inference is supported)

### 1. Clone the repository 

```bash
git clone https://github.com/ChristianLangridge/LEMBAS-RNN-benchmark.git
cd LEMBAS-RNN-benchmark
```

### 2. Create and activate a conda environment using lembasrnn_benchmark.yml 

```bash
conda env create -f lembasrnn_benchmark.yml 
```

> **Note for GPU support:** install the appropriate `torch` build for your CUDA version from [pytorch.org](https://pytorch.org)
>
>> *see lembasrnn_benchmark.yml for further dependecy specifications*

---

## Data Requirements 

This repository does **NOT** include the required data files (too large for version control). Can be sent on request using using the email badge listed at the top of the file. 

| File | Description | Expected Shape |
|---|---|---|
| `TF(full).tsv` | TF expression matrix (samples × TFs) | ~15,935 × 1,197 |
| `Geneexpression (full).tsv` | Target gene expression (samples × genes) | ~15,935 × 16,100 |
| `network(full).tsv` | Biological regulatory network (TF, Gene, Interaction) | ~1,153,904 × 3 |
| `Liver_bulk_external.tsv` | External validation cohort | ~262 x 16,100 |

### Network File Format

`network(full).tsv` must have exactly these three columns:

```
TF          Gene        Interaction
TP53        CDKN1A      1
MYC         CDK4        1
BRCA1       PTEN        -1
...
```

`Interaction` must be numeric (`1` = activation, `-1` = inhibition).

---

## Usage

### 1. Modifying `data_config.json`

Before running scripts, modify the data_config.json file with the absolute path to the data folder requested via email badge. 

```python
{
    "DATA_ROOT": "/path/to/your/data/directory"
}
```

### 2. Data Preprocessing

All models share the same preprocessing boilerplate. Run this first to generate `x_train`, `x_test`, `y_train`, `y_test`:

```python
%run "$REPO_ROOT/run/data preprocessing/model_boilerplate_remote.py"
```

This script:
- Loads TF and gene expression data
- Filters TFs to those present in the biological network (ensures all models use identical features)
- Applies an 80/20 train/test split with `random_state=888`

### 3. Training the Baseline Models

**MLR:**
```bash
jupyter notebook "run/model scripts/MLR/MLR.ipynb"
```

**XGBRF** (trains 16,100 targets in batches of 1,000 — expect ~20–60 min on a multi-core machine):
```bash
jupyter notebook "run/model scripts/XGBRF/XGBRF.ipynb"
```

### 4. Loading and Testing the RNN

The RNN is loaded from a saved checkpoint using `load_model_from_checkpoint()`:

```python
sys.path.insert(0, f"{REPO_ROOT}/run/model scripts/LEMBAS-RNN/")
from RNN_reconstructor import load_model_from_checkpoint

rnn_model = load_model_from_checkpoint(
    checkpoint_path=f'{MODELS_BASE_PATH}/RNN/uncentered_data_RNN/signaling_model.v1.pt',
    net_path=f'{DATA_ROOT}/Full data files/network(full).tsv',
    X_in_df=pd.DataFrame(x_validation),
    y_out_df=pd.DataFrame(y_validation),
    device='cpu',
    use_exact_training_params=True
)
```

Full inference walkthrough is in `run/model scripts/LEMBAS-RNN/RNN_reconstructor.py`.

### 5. Generating Benchmark Figures

```bash
# Training set fit (Fig 1A/B)
jupyter notebook "run/figures/Model-fitting/Fig1(fitting).ipynb"

# Test set performance (Fig 2A)
jupyter notebook "run/figures/Model-testing/Fig1.ipynb"

# External validation (Fig 3A)
jupyter notebook "run/figures/Model-validation/Fig1(validation).ipynb"
```

### 6. Generating Predictions Programmatically

```bash
python config/predictions/model_train_test_predictions.py
```

This script loads all three trained models and saves prediction arrays for downstream analysis.

---

## SHAP Interpretability

SHAP (SHapley Additive exPlanations) analysis is computed for two clinically relevant liver genes — **ALB** (Albumin) and **AFP** (Alpha-fetoprotein) — across all three models using the external validation cohort.

| Model | SHAP Method | Notes |
|---|---|---|
| MLR | `LinearExplainer` | Exact, fast |
| XGBRF | `TreeExplainer` | Exact, fast |
| RNN | `GradientExplainer` | Approximate; `n_samples=25`, `background=50` for feasibility |

### Running SHAP Analysis

```bash
# Baseline models (MLR + XGBRF)
python config/SHAP/SHAP_generation_baseline.py

# RNN (slow — optimised for feasibility with reduced sample counts)
python config/SHAP/SHAP_generation_RNN.py

# Load, validate, and plot saved SHAP values
python config/SHAP/SHAP_value_test_load_plot.py
```

SHAP outputs are saved as `.npz` files containing per-gene SHAP arrays, expected values, and feature names. Waterfall plots are generated for each model × gene combination in a publication-quality 2×3 grid.

---

## Known Issues & Limitations

### ⚠️ Feature Count Importance (1197 TFs and 16,100 target genes)
The saved RNN checkpoint was trained on 1,197 TF features and 16,100 target genes and the `network(full).tsv` file is alligned with that. Loading the checkpoint against a different network file or with different input dimensions will raise:

```
RuntimeError: size mismatch for input_layer.weights
```

**Workaround:** Either retrain the RNN on new number of features (recommended for reproducibility) using a new, aligned network file, or filter only inputs that match `network(full).tsv` and fill missing feature columnns with zero values. See `RNN_testing.ipynb` Step 6 for a full diagnostic.

### ⚠️ No Automated Test Suite
There is currently no `tests/` directory or CI pipeline. Unit and integration tests are planned.

### RNN SHAP is Slow
GradientExplainer on the full RNN is computationally expensive due to the iterative steady-state forward pass. With 2 x NVIDIA GeForce RTX-5080 GPUs, runtime for all 262 samples was around 4 minutes. This is will be much slower on CPU. 

**Workaround:** Lower the number of `RNN_BACKGROUND_SAMPLES`, `RNN_TEST_SAMPLES` and `RNN_N_SAMPLES` will make lessen computational overhead for use on CPU. 


---

## Contributing

This is an active research project. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature'`)
4. Push and open a Pull Request

Please ensure any new scripts avoid hardcoded paths and include basic inline documentation.

---

## Citation

If you use this codebase or the LEMBAS-RNN architecture in your work, please cite:

```
Langridge, C. (2025–2026). LEMBAS-RNN Benchmarking Project.
Rotational project, King's College London, Zhang Lab.
https://github.com/ChristianLangridge/LEMBAS-RNN-benchmark
```

---

## License

This project is licensed under the BSD-3-Clause License. See [LICENSE](LICENSE) for details.

---

*Developed as part of a rotational PhD project at King's College London in collaboration with Dr. Cheng Zhang.*


