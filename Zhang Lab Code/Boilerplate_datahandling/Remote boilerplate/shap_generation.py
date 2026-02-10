"""
Gene-Specific SHAP with GradientExplainer for RNN (Subset 50)
=============================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

sys.path.append('/home/christianl/Zhang-Lab/Zhang Lab Code/Tuning/uncentered_RNN_tuning')
from RNN_reconstructor import load_model_from_checkpoint

# ============================================================================
# Configuration
# ============================================================================

GENES_OF_INTEREST = ['ALB', 'AFP']

OUTPUT_BASE_PATH = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved SHAP values/gene_specific'
MODELS_BASE_PATH = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models'
DATA_BASE_PATH = '/home/christianl/Zhang-Lab/Zhang Lab Data'

os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# XGBRF settings
XGBRF_BATCH_SIZE = 1000

# RNN settings
RNN_SUBSET_SIZE = 50 # Compute only for the first N instances
ENABLE_RNN = False  # computing remaining RNN SHAP values in seperate script 

# ============================================================================
# Utility Functions
# ============================================================================

def print_section(title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

def find_gene_index(gene_name, gene_columns):
    try:
        idx = list(gene_columns).index(gene_name)
        return idx
    except ValueError:
        return None

def extract_xgbrf_model_from_batches(batch_models, gene_idx, batch_size=1000):
    """Extract single gene model from batch-trained XGBRF ensemble."""
    batch_idx = gene_idx // batch_size
    within_batch_idx = gene_idx % batch_size
    
    if batch_idx >= len(batch_models):
        raise IndexError(f"Batch index {batch_idx} out of range")
    
    batch_model = batch_models[batch_idx]
    
    if not hasattr(batch_model, 'estimators_'):
        raise AttributeError(f"Batch model is not a MultiOutputRegressor")
    
    if within_batch_idx >= len(batch_model.estimators_):
        raise IndexError(f"Within-batch index {within_batch_idx} out of range")
    
    return batch_model.estimators_[within_batch_idx]

# ============================================================================
# STEP 1: Load Validation Data
# ============================================================================

print_section("STEP 1: Loading Validation Data")

validation_dataset = pd.read_csv(f'{DATA_BASE_PATH}/Full data files/Liver_bulk_external.tsv', 
                                 sep='\t', header=0, index_col=0)

# Generate x_validation
net = pd.read_csv(f'{DATA_BASE_PATH}/Full data files/network(full).tsv', sep='\t')
network_tfs = set(net['TF'].unique())
network_genes = set(net['Gene'].unique())
network_nodes = network_tfs | network_genes

tf_expression = pd.read_csv(f'{DATA_BASE_PATH}/Full data files/TF(full).tsv', 
                            sep='\t', header=0, index_col=0)

usable_features = [tf for tf in tf_expression.columns if tf in network_nodes]
present_features = [tf for tf in usable_features if tf in validation_dataset.columns]
missing_features = [tf for tf in usable_features if tf not in validation_dataset.columns]

x_validation = pd.DataFrame(index=validation_dataset.index)
for feature in present_features:
    x_validation[feature] = validation_dataset[feature]
for feature in missing_features:
    x_validation[feature] = 0
x_validation = x_validation[usable_features]

# Generate y_validation
gene_expression_ref = pd.read_csv(f"{DATA_BASE_PATH}/Full data files/Geneexpression (full).tsv", 
                                  sep="\t", header=0, index_col=0)
expected_features = gene_expression_ref.columns.tolist()
new_input = validation_dataset.copy()

for feature in expected_features:
    if feature not in new_input.columns:
        new_input[feature] = 0

y_validation = new_input[expected_features]

print(f"✓ x_validation shape: {x_validation.shape}")
print(f"✓ y_validation shape: {y_validation.shape}")

feature_names_list = usable_features

# ============================================================================
# STEP 2: Find Gene Indices
# ============================================================================

print_section("STEP 2: Locating Genes of Interest")

gene_indices = {}
gene_columns = y_validation.columns

for gene in GENES_OF_INTEREST:
    idx = find_gene_index(gene, gene_columns)
    if idx is not None:
        gene_indices[gene] = idx
        print(f"✓ Found {gene} at global index {idx}")
    else:
        print(f"✗ Warning: {gene} not found")

if not gene_indices:
    print("\n✗ ERROR: No genes found!")
    sys.exit(1)

# ============================================================================
# STEP 3: Load Models
# ============================================================================

print_section("STEP 3: Loading Trained Models")

# MLR
print("Loading MLR model...")
mlr_path = f'{MODELS_BASE_PATH}/MLR/MLR_v3/MLR_model_v4(uncentered[FINAL]).joblib'
mlr_model = joblib.load(mlr_path)
print(f"  ✓ MLR loaded")
print(f"  Coefficient shape: {mlr_model.coef_.shape}")

# XGBRF
print("\n" + "─"*80)
print("Loading XGBRF models (Batch Structure)")
print("─"*80)

xgbrf_path = f'{MODELS_BASE_PATH}/XGBRF/XGBRF_v5/all_models_batch_XGBRF[uncentered_REALFINAL].joblib'
xgbrf_batch_models = joblib.load(xgbrf_path)

print(f"  ✓ Ensemble loaded ({len(xgbrf_batch_models)} batches)")

gene_specific_xgb_models = {}

for gene in GENES_OF_INTEREST:
    if gene not in gene_indices:
        continue
    
    gene_idx = gene_indices[gene]
    batch_idx = gene_idx // XGBRF_BATCH_SIZE
    within_batch_idx = gene_idx // XGBRF_BATCH_SIZE
    
    print(f"\n  {gene}: batch {batch_idx}, position {within_batch_idx}")
    
    try:
        model = extract_xgbrf_model_from_batches(xgbrf_batch_models, gene_idx, XGBRF_BATCH_SIZE)
        gene_specific_xgb_models[gene] = model
        print(f"    ✓ Extracted")
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")

del xgbrf_batch_models
gc.collect()
print("\n  ✓ Batch models purged")

# RNN
rnn_model = None
if ENABLE_RNN:
    print("\n" + "─"*80)
    print("Loading RNN model (for GradientExplainer)")
    print("─"*80)
    
    # Load on CPU initially to match the loading function signature
    rnn_model = load_model_from_checkpoint(
        checkpoint_path=f'{MODELS_BASE_PATH}/RNN/uncentered_data_RNN/signaling_model.v1.pt',
        net_path=f'{DATA_BASE_PATH}/Full data files/network(full).tsv',
        X_in_df=pd.DataFrame(x_validation),
        y_out_df=pd.DataFrame(y_validation),
        device='cpu',
        use_exact_training_params=True
    )
    # Don't move to GPU yet - we do that per-gene to keep memory clean or generally below
    print("  ✓ RNN loaded (CPU)")
else:
    print("\n⊗ RNN DISABLED (set ENABLE_RNN=True to compute)")

# ============================================================================
# STEP 4: Compute SHAP Values
# ============================================================================

print_section("STEP 4: Computing Gene-Specific SHAP Values")

shap_results = {
    'metadata': {
        'genes': GENES_OF_INTEREST,
        'gene_indices': gene_indices,
        'n_samples': x_validation.shape[0],
        'n_features': x_validation.shape[1],
        'feature_names': feature_names_list,
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    },
    'MLR': {},
    'XGBRF': {},
    'RNN': {}
}

for gene_name in GENES_OF_INTEREST:
    if gene_name not in gene_indices:
        continue
    
    gene_idx = gene_indices[gene_name]
    
    print(f"\n{'─'*80}")
    print(f"Processing {gene_name} (index {gene_idx})")
    print(f"{'─'*80}")
    
    # ────────────────────────────────────────────────────────────────────────
    # MLR (unchanged)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{gene_name}] Computing MLR SHAP values...")
    try:
        from sklearn.linear_model import LinearRegression
        
        if hasattr(mlr_model, 'estimators_'):
            coef = mlr_model.estimators_[gene_idx].coef_
            intercept = mlr_model.estimators_[gene_idx].intercept_
        else:
            coef = mlr_model.coef_[gene_idx, :]
            intercept = mlr_model.intercept_[gene_idx]
        
        proxy_model = LinearRegression()
        proxy_model.coef_ = coef.reshape(1, -1) if coef.ndim == 1 else coef
        
        if hasattr(intercept, '__len__'):
            proxy_model.intercept_ = np.array([intercept[0] if len(intercept) > 0 else intercept])
        else:
            proxy_model.intercept_ = np.array([intercept])
        
        proxy_model.n_features_in_ = x_validation.shape[1]
        proxy_model._sklearn_version = "1.0.0"
        
        test_pred = proxy_model.predict(x_validation.iloc[:5])
        print(f"  Test: mean {test_pred.mean():.4f}")
        
        mlr_explainer = shap.LinearExplainer(proxy_model, x_validation)
        mlr_shap_gene = mlr_explainer.shap_values(x_validation)
        
        if mlr_shap_gene.ndim == 3:
            mlr_shap_gene = mlr_shap_gene[:, :, 0]
        
        mlr_expected_value = mlr_explainer.expected_value
        if hasattr(mlr_expected_value, '__len__'):
            mlr_expected_value = float(mlr_expected_value[0])
        
        shap_results['MLR'][gene_name] = {
            'shap_values': mlr_shap_gene,
            'expected_value': mlr_expected_value,
            'explainer_type': 'LinearExplainer'
        }
        
        print(f"  ✓ Shape: {mlr_shap_gene.shape}, Expected: {mlr_expected_value:.4f}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        shap_results['MLR'][gene_name] = None
    
    # ────────────────────────────────────────────────────────────────────────
    # XGBRF (unchanged)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{gene_name}] Computing XGBRF SHAP values...")
    
    if gene_name not in gene_specific_xgb_models:
        print(f"  ✗ Skipping")
        shap_results['XGBRF'][gene_name] = None
        continue
    
    try:
        specific_xgb_model = gene_specific_xgb_models[gene_name]
        
        test_pred = specific_xgb_model.predict(x_validation.iloc[:5].values)
        print(f"  Test: mean {test_pred.mean():.4f}")
        
        xgb_explainer = shap.TreeExplainer(specific_xgb_model)
        xgb_shap_values = xgb_explainer.shap_values(x_validation, check_additivity=False)
        
        if isinstance(xgb_shap_values, list):
            xgb_shap_values = xgb_shap_values[0]
        
        xgb_expected_value = xgb_explainer.expected_value
        if hasattr(xgb_expected_value, '__len__'):
            xgb_expected_value = float(xgb_expected_value[0])
        
        shap_results['XGBRF'][gene_name] = {
            'shap_values': xgb_shap_values,
            'expected_value': float(xgb_expected_value),
            'explainer_type': 'TreeExplainer'
        }
        
        print(f"  ✓ Shape: {xgb_shap_values.shape}, Expected: {xgb_expected_value:.4f}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        shap_results['XGBRF'][gene_name] = None
    
    # ────────────────────────────────────────────────────────────────────────
    # RNN (GradientExplainer - GPU ACCELERATED)
    # ────────────────────────────────────────────────────────────────────────
    if ENABLE_RNN:
        print(f"\n[{gene_name}] Computing RNN SHAP values with GradientExplainer...")
        try:
            # 1. Setup GPU Device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"  ✓ Hardware acceleration: {device.upper()}")
            
            # Move model to GPU and set to EVAL mode
            # Eval mode is critical: removes dropout randomness and ensures consistent gradients
            rnn_model = rnn_model.to(device)
            rnn_model.eval()

            # 2. Define Target Wrapper
            # The SignalingModel returns a tuple (Y_hat, Y_full).
            # GradientExplainer requires a callable that returns a single tensor.
            # We wrap the model to intercept the output and select the specific gene index.
            class GeneTargetWrapper(nn.Module):
                def __init__(self, model, target_idx):
                    super().__init__()
                    self.model = model
                    self.target_idx = target_idx
                    
                def forward(self, x):
                    # Forward pass returns (Y_hat, Y_full), we only need Y_hat
                    y_hat, _ = self.model(x)
                    # Return specific gene column shaped as (Batch, 1)
                    return y_hat[:, self.target_idx].unsqueeze(1)

            # 3. Create Background (Must be Small & Tensor)
            # We sample 25 instances to serve as the background expectation.
            # Using too many samples here causes the explosion in compute time.
            print("  Generating background summary (N=25)...")
            background_data = shap.sample(x_validation, 25) 
            background_tensor = torch.FloatTensor(background_data.values).to(device)

            # 4. Initialize GradientExplainer
            # We pass the wrapped model which now behaves like a standard regression network
            target_model = GeneTargetWrapper(rnn_model, gene_idx)
            rnn_explainer = shap.GradientExplainer(target_model, background_tensor)
            
            # 5. Prepare Input Data (Subset)
            x_subset = x_validation.iloc[:RNN_SUBSET_SIZE]
            x_subset_tensor = torch.FloatTensor(x_subset.values).to(device)
            
            # 6. Compute SHAP values
            print(f"  Computing gradients for {len(x_subset)} instances...")
            rnn_shap_values = rnn_explainer.shap_values(x_subset_tensor)
            
            # GradientExplainer returns a list of tensors (one per output). 
            # Since our wrapper outputs 1 column, the list has length 1.
            if isinstance(rnn_shap_values, list):
                rnn_shap_values = rnn_shap_values[0]
            
            # 7. Metadata & Storage
            # Note: GradientExplainer doesn't have a simple .expected_value like KernelExplainer
            # We approximate it by taking the mean prediction of the background
            with torch.no_grad():
                expected_value_tensor = target_model(background_tensor).mean()
                rnn_expected_value = float(expected_value_tensor.cpu().numpy())
            
            shap_results['RNN'][gene_name] = {
                'shap_values': rnn_shap_values, # Already numpy array if shap version is recent, else might need .cpu().numpy()
                'expected_value': rnn_expected_value,
                'explainer_type': 'GradientExplainer',
                'n_samples_computed': len(x_subset),
                'subset_indices': x_subset.index.tolist()
            }
            
            print(f"  ✓ Shape: {rnn_shap_values.shape}")
            
        except Exception as e:
            print(f"  ✗ RNN failed: {e}")
            import traceback
            traceback.print_exc()
            shap_results['RNN'][gene_name] = None
            
        finally:
            # Clean up GPU memory after this gene
            if 'x_subset_tensor' in locals(): del x_subset_tensor
            if 'background_tensor' in locals(): del background_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ============================================================================
# STEP 5: Save Results
# ============================================================================

print_section("STEP 5: Saving Results")

output_file = f'{OUTPUT_BASE_PATH}/gene_specific_shap_values_validation.npz'

save_dict = {
    'feature_names': np.array(feature_names_list),
    'x_validation': x_validation.values,
    'genes': np.array(GENES_OF_INTEREST),
    'gene_indices': json.dumps(gene_indices)
}

for model_name in ['MLR', 'XGBRF', 'RNN']:
    for gene_name in GENES_OF_INTEREST:
        if gene_name in shap_results[model_name] and shap_results[model_name][gene_name] is not None:
            prefix = f'{model_name}_{gene_name}'
            save_dict[f'{prefix}_shap_values'] = shap_results[model_name][gene_name]['shap_values']
            save_dict[f'{prefix}_expected_value'] = shap_results[model_name][gene_name]['expected_value']
            
            # Add note for RNN subset if relevant for downstream loading
            if model_name == 'RNN':
                 save_dict[f'{prefix}_subset_indices'] = np.array(shap_results['RNN'][gene_name]['subset_indices'])

np.savez_compressed(output_file, **save_dict)
print(f"✓ Saved to: {output_file}")

print_section("✓ Script Completed!")

print(f"Genes: {list(gene_indices.keys())}")
print(f"\nModels computed:")
for model_name in ['MLR', 'XGBRF', 'RNN']:
    success_count = sum(1 for g in GENES_OF_INTEREST 
                       if g in shap_results[model_name] 
                       and shap_results[model_name][g] is not None)
    print(f"  {model_name}: {success_count}/{len(GENES_OF_INTEREST)} genes")
