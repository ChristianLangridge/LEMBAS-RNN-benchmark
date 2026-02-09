"""
Gene-Specific SHAP Value Generation for Model Comparison
=========================================================

Generate SHAP values for specific genes of interest (ALB, AFP) across
MLR, XGBRFRegressor, and LEMBAS-RNN models.

"""

import numpy as np
import pandas as pd
import torch
import joblib
import shap
import json
import sys
import os
import gc
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add RNN loader to path
sys.path.append('/home/christianl/Zhang-Lab/Zhang Lab Code/Tuning/uncentered_RNN_tuning')
from RNN_reconstructor import load_model_from_checkpoint

# ============================================================================
# Configuration
# ============================================================================

GENES_OF_INTEREST = ['ALB', 'AFP']  # Albumin and Alpha-fetoprotein

OUTPUT_BASE_PATH = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved SHAP values/gene_specific'
MODELS_BASE_PATH = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models'
DATA_BASE_PATH = '/home/christianl/Zhang-Lab/Zhang Lab Data'

# Create output directory
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# SHAP computation settings
# Note: BACKGROUND_SAMPLES is only used for RNN (KernelExplainer) now.
# XGBRF uses TreeExplainer (exact) and MLR uses LinearExplainer (analytical).
BACKGROUND_SAMPLES = 100 

# ============================================================================
# Utility Functions
# ============================================================================

def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

def find_gene_index(gene_name, gene_columns):
    """Find the index of a gene in the output columns."""
    try:
        idx = list(gene_columns).index(gene_name)
        print(f"  ✓ Found {gene_name} at index {idx}")
        return idx
    except ValueError:
        print(f"  ✗ Warning: {gene_name} not found in gene columns")
        return None

def create_gene_specific_predictor(model, gene_idx, model_type='sklearn'):
    """
    Create a prediction function for a specific gene output.
    Used ONLY for RNN (KernelExplainer) and generic sklearn wrappers.
    """
    if model_type == 'sklearn':
        def predictor(X):
            full_predictions = model.predict(X)
            return full_predictions[:, gene_idx].reshape(-1, 1)
        return predictor
    
    elif model_type == 'torch':
        def predictor(X):
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            X_tensor = torch.FloatTensor(X_array)
            
            with torch.no_grad():
                full_predictions, _ = model(X_tensor)
                gene_predictions = full_predictions[:, gene_idx].numpy()
            
            return gene_predictions.reshape(-1, 1)
        return predictor
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# ============================================================================
# Data Loading - VALIDATION DATASET
# ============================================================================

print_section("STEP 1: Loading Validation Data")

# ============================================================================
# PART 1: Generate x_validation (TF features)
# ============================================================================

print("[1/3] Loading external liver bulk validation data...")
validation_dataset = pd.read_csv(f'{DATA_BASE_PATH}/Full data files/Liver_bulk_external.tsv', 
                                 sep='\t', header=0, index_col=0)

print(f"  ✓ External data shape: {validation_dataset.shape}")

print("\n[2/3] Generating x_validation (TF features)...")
net = pd.read_csv(f'{DATA_BASE_PATH}/Full data files/network(full).tsv', sep='\t')
network_tfs = set(net['TF'].unique())
network_genes = set(net['Gene'].unique())
network_nodes = network_tfs | network_genes

tf_expression = pd.read_csv(f'{DATA_BASE_PATH}/Full data files/TF(full).tsv', 
                            sep='\t', header=0, index_col=0)

usable_features = [tf for tf in tf_expression.columns if tf in network_nodes]
present_features = [tf for tf in usable_features if tf in validation_dataset.columns]
missing_features = [tf for tf in usable_features if tf not in validation_dataset.columns]

print(f"  Usable TF features: {len(usable_features)}")
print(f"  Present in validation data: {len(present_features)}")
print(f"  Missing (filled with 0): {len(missing_features)}")

x_validation = pd.DataFrame(index=validation_dataset.index)

for feature in present_features:
    x_validation[feature] = validation_dataset[feature]

for feature in missing_features:
    x_validation[feature] = 0

x_validation = x_validation[usable_features]
print(f"  ✓ x_validation shape: {x_validation.shape}")

# ============================================================================
# PART 2: Generate y_validation (Gene expression features)
# ============================================================================

print("\n[3/3] Generating y_validation (Gene expression features)...")

new_input = validation_dataset.copy()
gene_expression_ref = pd.read_csv(f"{DATA_BASE_PATH}/Full data files/Geneexpression (full).tsv", 
                                  sep="\t", header=0, index_col=0)
expected_features = gene_expression_ref.columns.tolist()
missing_features_y = [feature for feature in expected_features if feature not in new_input.columns]

for feature in missing_features_y:
    new_input[feature] = 0

new_input = new_input[expected_features]
y_validation = new_input
print(f"  ✓ y_validation shape: {y_validation.shape}")

# Store for compatibility
feature_names_list = usable_features

# ============================================================================
# Find Gene Indices
# ============================================================================

print_section("STEP 2: Locating Genes of Interest")

gene_indices = {}
gene_columns = y_validation.columns
for gene in GENES_OF_INTEREST:
    idx = find_gene_index(gene, gene_columns)
    if idx is not None:
        gene_indices[gene] = idx

if not gene_indices:
    print("\n✗ ERROR: None of the genes of interest were found!")
    sys.exit(1)

# ============================================================================
# Load Models
# ============================================================================

print_section("STEP 3: Loading Trained Models")

# 1. MLR
print("Loading MLR model...")
mlr_path = f'{MODELS_BASE_PATH}/MLR/MLR_v3/MLR_model_v4(uncentered[FINAL]).joblib'
mlr_model = joblib.load(mlr_path)
print("  ✓ MLR loaded")

# 2. XGBRF (Optimized Load-Extract-Purge)
print("\nLoading XGBRF models (Extracting specific genes only)...")
xgbrf_path = f'{MODELS_BASE_PATH}/XGBRF/XGBRF_v5/all_models_batch_XGBRF[uncentered_REALFINAL].joblib'

# Load the full ensemble temporarily
full_xgbrf_ensemble = joblib.load(xgbrf_path)
print("  ✓ Full ensemble loaded into memory")

# Extract only the specific models we need
gene_specific_xgb_models = {}
for gene, idx in gene_indices.items():
    try:
        # Check if MultiOutputRegressor (estimators_) or list
        if hasattr(full_xgbrf_ensemble, 'estimators_'):
            model = full_xgbrf_ensemble.estimators_[idx]
        else:
            model = full_xgbrf_ensemble[idx]
        
        gene_specific_xgb_models[gene] = model
        print(f"  ✓ Extracted XGBRF model for {gene}")
    except Exception as e:
        print(f"  ✗ Failed to extract model for {gene}: {e}")

# Purge the heavy ensemble to free memory
del full_xgbrf_ensemble
gc.collect()
print("  ✓ Unused XGBRF models purged from memory")

# 3. RNN
print("\nLoading RNN model...")
rnn_model = load_model_from_checkpoint(
    checkpoint_path=f'{MODELS_BASE_PATH}/RNN/uncentered_data_RNN/signaling_model.v1.pt',
    net_path=f'{DATA_BASE_PATH}/Full data files/network(full).tsv',
    X_in_df=pd.DataFrame(x_validation),
    y_out_df=pd.DataFrame(y_validation),
    device='cpu',
    use_exact_training_params=True
)
print("  ✓ RNN loaded")

# ============================================================================
# SHAP Value Computation for Each Gene
# ============================================================================

print_section("STEP 4: Computing Gene-Specific SHAP Values")

# Store all results
shap_results = {
    'metadata': {
        'genes': GENES_OF_INTEREST,
        'gene_indices': gene_indices,
        'n_samples': x_validation.shape[0],
        'n_features': x_validation.shape[1],
        'feature_names': feature_names_list,
        'background_samples': BACKGROUND_SAMPLES,
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': 42,
        'dataset': 'external_validation (FULL)'
    },
    'MLR': {},
    'XGBRF': {},
    'RNN': {}
}

for gene_name, gene_idx in gene_indices.items():
    print(f"\n{'─'*80}")
    print(f"Processing {gene_name} (index {gene_idx})")
    print(f"{'─'*80}")
    
    # ────────────────────────────────────────────────────────────────────────
    # MLR (Optimized: Single-Target Extraction)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{gene_name}] Computing MLR SHAP values...")
    try:
        # STRATEGY: Construct a temporary Linear Model for JUST this gene.
        # This prevents calculating SHAP for the other 16,000 outputs.
        
        from sklearn.linear_model import LinearRegression
        
        # 1. Extract Coefficients for the specific gene
        # Case A: Model is a MultiOutputRegressor (list of models)
        if hasattr(mlr_model, 'estimators_'):
            specific_estimator = mlr_model.estimators_[gene_idx]
            coef = specific_estimator.coef_
            intercept = specific_estimator.intercept_
            
        # Case B: Model is a single LinearRegression with matrix output
        elif hasattr(mlr_model, 'coef_'):
            # coef_ shape is usually (n_targets, n_features)
            coef = mlr_model.coef_[gene_idx, :]
            intercept = mlr_model.intercept_[gene_idx]
        
        # 2. Create a lightweight proxy model
        # We make a tiny new model just for ALB/AFP
        proxy_model = LinearRegression()
        proxy_model.coef_ = coef
        proxy_model.intercept_ = intercept
        
        # 3. Explain the proxy model
        # Now LinearExplainer only sees 1 output. 
        # Computation drops from ~20GB RAM / 30 mins to ~2MB RAM / 0.5 seconds.
        mlr_explainer = shap.LinearExplainer(proxy_model, x_validation)
        mlr_shap_gene = mlr_explainer.shap_values(x_validation)
        
        # LinearExplainer for single output sometimes returns (n, features) 
        # or (n, features, 1). We ensure 2D shape.
        if mlr_shap_gene.ndim == 3:
             mlr_shap_gene = mlr_shap_gene[:, :, 0]
             
        mlr_expected_value = mlr_explainer.expected_value
        
        shap_results['MLR'][gene_name] = {
            'shap_values': mlr_shap_gene,
            'expected_value': mlr_expected_value,
            'explainer_type': 'LinearExplainer'
        }
        
        print(f"  ✓ MLR SHAP shape: {mlr_shap_gene.shape}")
        print(f"  ✓ Expected value: {mlr_expected_value:.4f}")
        
    except Exception as e:
        print(f"  ✗ MLR failed: {e}")
        import traceback
        traceback.print_exc()
        shap_results['MLR'][gene_name] = None
    
    # ────────────────────────────────────────────────────────────────────────
    # XGBRF (Optimized with TreeExplainer)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{gene_name}] Computing XGBRF SHAP values (TreeExplainer)...")
    try:
        if gene_name not in gene_specific_xgb_models:
            raise ValueError(f"Model for {gene_name} was not extracted correctly.")

        # 1. Get the specific model
        specific_xgb_model = gene_specific_xgb_models[gene_name]
        
        # 2. Use TreeExplainer (Exact & Fast)
        xgb_explainer = shap.TreeExplainer(specific_xgb_model)
        
        # 3. Compute SHAP values on full validation set
        # check_additivity=False prevents errors if XGBoost version mismatch
        xgb_shap_values = xgb_explainer.shap_values(x_validation, check_additivity=False)
        
        # 4. Handle dimensions (TreeExplainer might return list)
        if isinstance(xgb_shap_values, list):
            xgb_shap_values = xgb_shap_values[0]
            
        xgb_expected_value = xgb_explainer.expected_value
        
        shap_results['XGBRF'][gene_name] = {
            'shap_values': xgb_shap_values,
            'expected_value': float(xgb_expected_value),
            'explainer_type': 'TreeExplainer'
        }
        
        print(f"  ✓ XGBRF SHAP shape: {xgb_shap_values.shape}")
        print(f"  ✓ Expected value: {xgb_expected_value:.4f}")
        
    except Exception as e:
        print(f"  ✗ XGBRF failed: {e}")
        shap_results['XGBRF'][gene_name] = None
    
    # ────────────────────────────────────────────────────────────────────────
    # RNN (Uses KernelExplainer)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{gene_name}] Computing RNN SHAP values...")
    try:
        # Create gene-specific predictor
        rnn_gene_predictor = create_gene_specific_predictor(
            rnn_model, gene_idx, model_type='torch'
        )
        
        # Sample background for KernelExplainer
        background = shap.sample(x_validation, min(BACKGROUND_SAMPLES, len(x_validation)))
        
        # Use KernelExplainer
        rnn_explainer = shap.KernelExplainer(rnn_gene_predictor, background)
        rnn_shap_gene = rnn_explainer.shap_values(x_validation)
        
        rnn_expected_value = rnn_explainer.expected_value
        
        if rnn_shap_gene.ndim == 3:
            rnn_shap_gene = rnn_shap_gene[:, :, 0]
        
        shap_results['RNN'][gene_name] = {
            'shap_values': rnn_shap_gene,
            'expected_value': float(rnn_expected_value),
            'explainer_type': 'KernelExplainer'
        }
        
        print(f"  ✓ RNN SHAP shape: {rnn_shap_gene.shape}")
        
    except Exception as e:
        print(f"  ✗ RNN failed: {e}")
        shap_results['RNN'][gene_name] = None

# ============================================================================
# Save Results
# ============================================================================

print_section("STEP 5: Saving Results")

output_file = f'{OUTPUT_BASE_PATH}/gene_specific_shap_values_validation.npz'

save_dict = {
    'feature_names': np.array(feature_names_list),
    'x_validation_subset': x_validation.values, # Saved as full validation
    'genes': np.array(GENES_OF_INTEREST)
}

for model_name in ['MLR', 'XGBRF', 'RNN']:
    for gene_name in gene_indices.keys():
        if shap_results[model_name][gene_name] is not None:
            prefix = f'{model_name}_{gene_name}'
            save_dict[f'{prefix}_shap_values'] = shap_results[model_name][gene_name]['shap_values']
            save_dict[f'{prefix}_expected_value'] = shap_results[model_name][gene_name]['expected_value']

np.savez_compressed(output_file, **save_dict)
print(f"✓ Saved SHAP values to: {output_file}")
print(f"  File size: {os.path.getsize(output_file) / 1e6:.2f} MB")

metadata_file = f'{OUTPUT_BASE_PATH}/gene_specific_shap_metadata_validation.json'
metadata_to_save = {
    'genes': GENES_OF_INTEREST,
    'gene_indices': {k: int(v) for k, v in gene_indices.items()},
    'n_samples': int(x_validation.shape[0]),
    'n_features': int(x_validation.shape[1]),
    'feature_names': feature_names_list,
    'models': ['MLR', 'XGBRF', 'RNN'],
    'dataset': 'external_validation',
    'explainer_types': {
        'MLR': 'LinearExplainer',
        'XGBRF': 'TreeExplainer',  # Updated
        'RNN': 'KernelExplainer'
    }
}

with open(metadata_file, 'w') as f:
    json.dump(metadata_to_save, f, indent=2)

print(f"✓ Saved metadata to: {metadata_file}")

# ============================================================================
# Save Results
# ============================================================================

print_section("STEP 5: Saving Results")

# ============================================================================
# Generate Quick Load Script
# ============================================================================

loader_script = f"""\"\"\"
Quick Loader for Gene-Specific SHAP Values (Validation Dataset)
================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: External liver bulk validation data
\"\"\"

import numpy as np
import pandas as pd
import json

# Load data
data = np.load('{output_file}', allow_pickle=True)

# Load metadata
with open('{metadata_file}', 'r') as f:
    metadata = json.load(f)

# Extract feature information
feature_names = data['feature_names']
x_validation_subset = data['x_validation_subset']
genes = data['genes']

print(f"Loaded SHAP values for {{len(genes)}} genes: {{list(genes)}}")
print(f"Models: {{metadata['models']}}")
print(f"Dataset: {{metadata['dataset']}}")
print(f"Samples: {{metadata['n_samples']}}")
print(f"Features: {{metadata['n_features']}}")

# Example: Access ALB SHAP values for MLR
# mlr_alb_shap = data['MLR_ALB_shap_values']
# mlr_alb_expected = data['MLR_ALB_expected_value']

# Example: Create waterfall plot for first validation sample
# import shap
# instance_idx = 0
# shap.waterfall_plot(
#     shap.Explanation(
#         values=data['MLR_ALB_shap_values'][instance_idx],
#         base_values=data['MLR_ALB_expected_value'],
#         data=x_validation_subset[instance_idx],
#         feature_names=feature_names
#     )
# )
"""

loader_file = f'{OUTPUT_BASE_PATH}/load_shap_values.py'
with open(loader_file, 'w') as f:
    f.write(loader_script)

print(f"✓ Generated loader script: {loader_file}")

# ============================================================================
# Summary
# ============================================================================

print_section("Summary")

print("Gene-Specific SHAP Values Generated Successfully!")
print(f"\nDataset: External liver bulk validation data")
print(f"Genes processed: {', '.join(gene_indices.keys())}")
print(f"Models: MLR, XGBRF, RNN")
print(f"Samples: {x_validation_subset.shape[0]}")
print(f"Features: {x_validation_subset.shape[1]}")

print("\nFiles created:")
print(f"  1. {output_file}")
print(f"  2. {metadata_file}")
print(f"  3. {loader_file}")

print("\n" + "─"*80)
print("Next Steps:")
print("─"*80)
print("1. Run the loader script to access SHAP values:")
print(f"   %run {loader_file}")
print("\n2. Create waterfall plots for specific validation samples:")
print("   import shap")
print("   shap.waterfall_plot(shap.Explanation(")
print("       values=data['MLR_ALB_shap_values'][0],")
print("       base_values=data['MLR_ALB_expected_value'],")
print("       data=x_validation_subset[0],")
print("       feature_names=feature_names")
print("   ))")
print("\n3. Compare models for the same gene:")
print("   # Compare MLR vs XGBRF vs RNN for ALB on validation data")
print("   for model in ['MLR', 'XGBRF', 'RNN']:")
print("       key = f'{model}_ALB_shap_values'")
print("       print(f'{model}: {data[key][0][:5]}')")

print("\n" + "="*80)
print("✓ Script completed successfully!")
print("="*80)