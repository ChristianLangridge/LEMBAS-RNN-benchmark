"""
Gene-Specific SHAP Value Generation - FINAL CORRECTED VERSION
==============================================================

FIXES:
1. MLR: Proper sklearn proxy model initialization with 2D reshaping
2. XGBRF: Correct batch-based extraction (17 batches × 1000 genes each)
3. RNN: Reduced computational budget with staged testing

This version correctly handles the XGBRF batch training structure.
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

# XGBRF batch structure
XGBRF_BATCH_SIZE = 1000  # As used in training

# RNN settings (reduced for testing)
BACKGROUND_SAMPLES_RNN = 50
RNN_TEST_SAMPLES = 50
ENABLE_RNN = False  # Enable after MLR/XGBRF validated

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
    """
    Extract a single gene's model from batch-trained XGBRF ensemble.
    
    Args:
        batch_models: List of MultiOutputRegressor objects
        gene_idx: Global gene index (0-16099)
        batch_size: Number of genes per batch (default 1000)
    
    Returns:
        Single XGBRFRegressor model for the specified gene
    """
    batch_idx = gene_idx // batch_size
    within_batch_idx = gene_idx % batch_size
    
    if batch_idx >= len(batch_models):
        raise IndexError(f"Batch index {batch_idx} out of range (max {len(batch_models)-1})")
    
    batch_model = batch_models[batch_idx]
    
    if not hasattr(batch_model, 'estimators_'):
        raise AttributeError(f"Batch model at index {batch_idx} is not a MultiOutputRegressor")
    
    if within_batch_idx >= len(batch_model.estimators_):
        raise IndexError(f"Within-batch index {within_batch_idx} out of range for batch {batch_idx}")
    
    return batch_model.estimators_[within_batch_idx]

def create_gene_specific_predictor(model, gene_idx):
    """Create RNN prediction function for specific gene."""
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

# ============================================================================
# STEP 1: Load Validation Data
# ============================================================================

print_section("STEP 1: Loading Validation Data")

validation_dataset = pd.read_csv(f'{DATA_BASE_PATH}/Full data files/Liver_bulk_external.tsv', 
                                 sep='\t', header=0, index_col=0)

# Generate x_validation (TF features)
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

# Generate y_validation (Gene expression features)
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
        print(f"✗ Warning: {gene} not found in gene columns")

if not gene_indices:
    print("\n✗ ERROR: None of the genes of interest were found!")
    sys.exit(1)

# ============================================================================
# STEP 3: Load Models
# ============================================================================

print_section("STEP 3: Loading Trained Models")

# ────────────────────────────────────────────────────────────────────────────
# MLR
# ────────────────────────────────────────────────────────────────────────────
print("Loading MLR model...")
mlr_path = f'{MODELS_BASE_PATH}/MLR/MLR_v3/MLR_model_v4(uncentered[FINAL]).joblib'
mlr_model = joblib.load(mlr_path)
print(f"  ✓ MLR loaded")
print(f"  Coefficient shape: {mlr_model.coef_.shape}")
print(f"  Intercept shape: {mlr_model.intercept_.shape}")

# ────────────────────────────────────────────────────────────────────────────
# XGBRF (Batch-Based Extraction)
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*80)
print("Loading XGBRF models (Batch Structure)")
print("─"*80)

xgbrf_path = f'{MODELS_BASE_PATH}/XGBRF/XGBRF_v5/all_models_batch_XGBRF[uncentered_REALFINAL].joblib'
xgbrf_batch_models = joblib.load(xgbrf_path)

print(f"  ✓ Ensemble loaded")
print(f"  Structure: List of {len(xgbrf_batch_models)} batch models")
print(f"  Batch size: {XGBRF_BATCH_SIZE} genes per batch")
print(f"  Total gene capacity: {len(xgbrf_batch_models) * XGBRF_BATCH_SIZE}")

# Extract specific gene models using batch logic
gene_specific_xgb_models = {}

for gene in GENES_OF_INTEREST:
    if gene not in gene_indices:
        print(f"\n  {gene}: Skipping (not found in gene columns)")
        continue
    
    gene_idx = gene_indices[gene]
    batch_idx = gene_idx // XGBRF_BATCH_SIZE
    within_batch_idx = gene_idx % XGBRF_BATCH_SIZE
    
    print(f"\n  {gene}:")
    print(f"    Global gene index: {gene_idx}")
    print(f"    Batch index: {batch_idx}")
    print(f"    Within-batch index: {within_batch_idx}")
    
    try:
        model = extract_xgbrf_model_from_batches(xgbrf_batch_models, gene_idx, XGBRF_BATCH_SIZE)
        gene_specific_xgb_models[gene] = model
        print(f"    ✓ Successfully extracted model")
        
        # Test prediction
        dummy_X = np.random.randn(5, x_validation.shape[1])
        pred = model.predict(dummy_X)
        print(f"    ✓ Test prediction successful: shape {pred.shape}")
        
    except Exception as e:
        print(f"    ✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()

# Cleanup
del xgbrf_batch_models
gc.collect()
print("\n  ✓ Batch models purged from memory")

# ────────────────────────────────────────────────────────────────────────────
# RNN (Optional)
# ────────────────────────────────────────────────────────────────────────────
rnn_model = None
if ENABLE_RNN:
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
        'xgbrf_batch_size': XGBRF_BATCH_SIZE,
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    },
    'MLR': {},
    'XGBRF': {},
    'RNN': {}
}

for gene_name in GENES_OF_INTEREST:
    if gene_name not in gene_indices:
        print(f"\n✗ Skipping {gene_name} - not found in gene columns")
        continue
    
    gene_idx = gene_indices[gene_name]
    
    print(f"\n{'─'*80}")
    print(f"Processing {gene_name} (global index {gene_idx})")
    print(f"{'─'*80}")
    
    # ────────────────────────────────────────────────────────────────────────
    # MLR (Fixed sklearn proxy initialization)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{gene_name}] Computing MLR SHAP values...")
    try:
        from sklearn.linear_model import LinearRegression
        
        # Extract coefficients for this gene
        if hasattr(mlr_model, 'estimators_'):
            specific_estimator = mlr_model.estimators_[gene_idx]
            coef = specific_estimator.coef_
            intercept = specific_estimator.intercept_
        else:
            coef = mlr_model.coef_[gene_idx, :]
            intercept = mlr_model.intercept_[gene_idx]
        
        # Create properly initialized proxy model
        proxy_model = LinearRegression()
        
        # FIX: Ensure 2D shape (1, n_features)
        proxy_model.coef_ = coef.reshape(1, -1) if coef.ndim == 1 else coef
        
        # FIX: Ensure intercept is array
        if hasattr(intercept, '__len__'):
            proxy_model.intercept_ = np.array([intercept[0] if len(intercept) > 0 else intercept])
        else:
            proxy_model.intercept_ = np.array([intercept])
        
        # Set required sklearn metadata
        proxy_model.n_features_in_ = x_validation.shape[1]
        proxy_model._sklearn_version = "1.0.0"
        
        # Validate with test prediction
        test_pred = proxy_model.predict(x_validation.iloc[:5])
        print(f"  Test prediction: shape {test_pred.shape}, mean {test_pred.mean():.4f}")
        
        # Compute SHAP with LinearExplainer
        print(f"  Creating LinearExplainer...")
        mlr_explainer = shap.LinearExplainer(proxy_model, x_validation)
        
        print(f"  Computing SHAP values...")
        mlr_shap_gene = mlr_explainer.shap_values(x_validation)
        
        # Handle dimensionality
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
        
        print(f"  ✓ MLR SHAP shape: {mlr_shap_gene.shape}")
        print(f"  ✓ Expected value: {mlr_expected_value:.4f}")
        print(f"  ✓ Mean |SHAP|: {np.abs(mlr_shap_gene).mean():.6f}")
        
    except Exception as e:
        print(f"  ✗ MLR failed: {e}")
        import traceback
        traceback.print_exc()
        shap_results['MLR'][gene_name] = None
    
    # ────────────────────────────────────────────────────────────────────────
    # XGBRF (Using extracted batch model)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[{gene_name}] Computing XGBRF SHAP values...")
    
    if gene_name not in gene_specific_xgb_models:
        print(f"  ✗ Skipping - model extraction failed")
        shap_results['XGBRF'][gene_name] = None
        continue
    
    try:
        specific_xgb_model = gene_specific_xgb_models[gene_name]
        
        # Test prediction first
        test_pred = specific_xgb_model.predict(x_validation.iloc[:5].values)
        print(f"  Test prediction: shape {test_pred.shape}, mean {test_pred.mean():.4f}")
        
        # Use TreeExplainer (exact and fast)
        print(f"  Creating TreeExplainer...")
        xgb_explainer = shap.TreeExplainer(specific_xgb_model)
        
        print(f"  Computing SHAP values...")
        xgb_shap_values = xgb_explainer.shap_values(x_validation, check_additivity=False)
        
        # Handle list return (some SHAP versions)
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
        
        print(f"  ✓ XGBRF SHAP shape: {xgb_shap_values.shape}")
        print(f"  ✓ Expected value: {xgb_expected_value:.4f}")
        print(f"  ✓ Mean |SHAP|: {np.abs(xgb_shap_values).mean():.6f}")
        
    except Exception as e:
        print(f"  ✗ XGBRF failed: {e}")
        import traceback
        traceback.print_exc()
        shap_results['XGBRF'][gene_name] = None
    
    # ────────────────────────────────────────────────────────────────────────
    # RNN (KernelExplainer with reduced samples)
    # ────────────────────────────────────────────────────────────────────────
    if ENABLE_RNN:
        print(f"\n[{gene_name}] Computing RNN SHAP values...")
        print(f"  Background samples: {BACKGROUND_SAMPLES_RNN}")
        print(f"  Test samples: {RNN_TEST_SAMPLES}")
        
        try:
            rnn_gene_predictor = create_gene_specific_predictor(rnn_model, gene_idx)
            
            # Test prediction
            test_pred = rnn_gene_predictor(x_validation.iloc[:5].values)
            print(f"  Test prediction: shape {test_pred.shape}, mean {test_pred.mean():.4f}")
            
            # Reduce computation
            x_val_subset = x_validation.iloc[:RNN_TEST_SAMPLES]
            background = shap.sample(x_validation, min(BACKGROUND_SAMPLES_RNN, len(x_validation)))
            
            print(f"  Creating KernelExplainer...")
            rnn_explainer = shap.KernelExplainer(rnn_gene_predictor, background)
            
            print(f"  Computing SHAP values (may take 10-30 minutes)...")
            rnn_shap_gene = rnn_explainer.shap_values(x_val_subset, silent=False)
            
            rnn_expected_value = rnn_explainer.expected_value
            if hasattr(rnn_expected_value, '__len__'):
                rnn_expected_value = float(rnn_expected_value[0])
            
            if rnn_shap_gene.ndim == 3:
                rnn_shap_gene = rnn_shap_gene[:, :, 0]
            
            shap_results['RNN'][gene_name] = {
                'shap_values': rnn_shap_gene,
                'expected_value': float(rnn_expected_value),
                'explainer_type': 'KernelExplainer',
                'n_samples_computed': RNN_TEST_SAMPLES
            }
            
            print(f"  ✓ RNN SHAP shape: {rnn_shap_gene.shape}")
            print(f"  ✓ Expected value: {rnn_expected_value:.4f}")
            print(f"  ✓ Mean |SHAP|: {np.abs(rnn_shap_gene).mean():.6f}")
            
        except Exception as e:
            print(f"  ✗ RNN failed: {e}")
            import traceback
            traceback.print_exc()
            shap_results['RNN'][gene_name] = None

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

np.savez_compressed(output_file, **save_dict)
print(f"✓ Saved SHAP values to: {output_file}")
print(f"  File size: {os.path.getsize(output_file) / 1e6:.2f} MB")

# Save metadata
metadata_file = f'{OUTPUT_BASE_PATH}/gene_specific_shap_metadata_validation.json'
metadata_to_save = {
    'genes': GENES_OF_INTEREST,
    'gene_indices': {k: int(v) for k, v in gene_indices.items()},
    'n_samples': int(x_validation.shape[0]),
    'n_features': int(x_validation.shape[1]),
    'feature_names': feature_names_list,
    'xgbrf_batch_size': XGBRF_BATCH_SIZE,
    'models': ['MLR', 'XGBRF'] + (['RNN'] if ENABLE_RNN else []),
    'explainer_types': {
        'MLR': 'LinearExplainer',
        'XGBRF': 'TreeExplainer',
        'RNN': 'KernelExplainer' if ENABLE_RNN else 'Not computed'
    },
    'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(metadata_file, 'w') as f:
    json.dump(metadata_to_save, f, indent=2)
print(f"✓ Saved metadata to: {metadata_file}")

# ============================================================================
# Summary
# ============================================================================

print_section("✓ Script Completed Successfully!")

print(f"Genes processed: {list(gene_indices.keys())}")
print(f"\nModels computed:")
for model_name in ['MLR', 'XGBRF', 'RNN']:
    success_count = sum(1 for g in GENES_OF_INTEREST 
                       if g in shap_results[model_name] 
                       and shap_results[model_name][g] is not None)
    print(f"  {model_name}: {success_count}/{len(GENES_OF_INTEREST)} genes")

print(f"\nOutput files:")
print(f"  1. {output_file}")
print(f"  2. {metadata_file}")

print("\n" + "─"*80)
print("Next Steps:")
print("─"*80)
print("1. Run verify_shap_outputs.py to validate and visualize results")
print("2. If satisfied, set ENABLE_RNN=True to compute RNN SHAP values")
print("3. Create publication-quality waterfall/beeswarm plots")
print("4. Compare feature importance across models")
