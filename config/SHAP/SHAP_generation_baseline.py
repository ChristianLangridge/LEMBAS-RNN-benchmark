import numpy as np
import pandas as pd
import torch
import joblib
import shap
import sys
import gc
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json

# Resolve REPO_ROOT and DATA_ROOT
# Works whether the script is run directly (python script.py) 
# or via %run from a notebook
if 'REPO_ROOT' not in dir():
    _root = next(p for p in Path(__file__).resolve().parents if (p / "README.md").exists())
    REPO_ROOT = str(_root)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if 'DATA_ROOT' not in dir():
    with open(Path(REPO_ROOT) / "data_config.json", "r") as f:
        DATA_ROOT = json.load(f)["DATA_ROOT"]

sys.path.insert(0, f"{REPO_ROOT}/config/SHAP/")
from RNN_reconstructor import load_model_from_checkpoint

# ============================================================================
# Configuration
# ============================================================================

GENES_OF_INTEREST = ['ALB', 'AFP']

OUTPUT_BASE_PATH = f"{DATA_ROOT}/Saved SHAP values/gene_specific"
MODELS_BASE_PATH = f"{DATA_ROOT}/Saved models"
DATA_BASE_PATH = DATA_ROOT

os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# XGBRF settings
XGBRF_BATCH_SIZE = 1000

# RNN settings (DeepExplainer is much faster - can use full dataset!)
BACKGROUND_SAMPLES_RNN = 20   # DeepExplainer needs far fewer samples
RNN_COMPUTE_ALL_SAMPLES = True  # Can now compute on all 262 samples (fast!)
ENABLE_RNN = False  # Set to True when ready

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
        print(f"✗ Warning: {gene} not found")

if not gene_indices:
    print("\n✗ ERROR: No genes found!")
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
# XGBRF (Batch Structure)
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

# Extract specific gene models
gene_specific_xgb_models = {}

for gene in GENES_OF_INTEREST:
    if gene not in gene_indices:
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
        
        # Test prediction
        dummy_X = np.random.randn(5, x_validation.shape[1])
        pred = model.predict(dummy_X)
        print(f"    ✓ Successfully extracted model")
        print(f"    ✓ Test prediction successful: shape {pred.shape}")
        
    except Exception as e:
        print(f"    ✗ Extraction failed: {e}")

# Cleanup
del xgbrf_batch_models
gc.collect()
print("\n  ✓ Batch models purged from memory")

# ────────────────────────────────────────────────────────────────────────────
# RNN (if enabled)
# ────────────────────────────────────────────────────────────────────────────
rnn_model = None
if ENABLE_RNN:
    print("\n" + "─"*80)
    print("Loading RNN model (for DeepExplainer)")
    print("─"*80)
    
    rnn_model = load_model_from_checkpoint(
        checkpoint_path=f'{MODELS_BASE_PATH}/RNN/uncentered_data_RNN/signaling_model.v1.pt',
        net_path=f'{DATA_BASE_PATH}/Full data files/network(full).tsv',
        X_in_df=pd.DataFrame(x_validation),
        y_out_df=pd.DataFrame(y_validation),
        device='cpu',
        use_exact_training_params=True
    )
    rnn_model.eval()  # Important for DeepExplainer
    print("  ✓ RNN loaded and set to eval mode")
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
        'rnn_background_samples': BACKGROUND_SAMPLES_RNN if ENABLE_RNN else None,
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
    # MLR (LinearExplainer - unchanged)
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
        
        # Ensure 2D shape (1, n_features)
        proxy_model.coef_ = coef.reshape(1, -1) if coef.ndim == 1 else coef
        
        # Ensure intercept is array
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
    # XGBRF (TreeExplainer - unchanged)
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
    # RNN (DeepExplainer - NEW!)
    # ────────────────────────────────────────────────────────────────────────
    if ENABLE_RNN:
        print(f"\n[{gene_name}] Computing RNN SHAP values with DeepExplainer...")
        try:
            # Prepare background data (DeepExplainer needs much less!)
            background_data = shap.sample(x_validation, BACKGROUND_SAMPLES_RNN)
            background_tensor = torch.FloatTensor(background_data.values)
            
            print(f"  Background samples: {BACKGROUND_SAMPLES_RNN}")
            
            # Prepare test data
            if RNN_COMPUTE_ALL_SAMPLES:
                test_data = x_validation
                print(f"  Computing SHAP for all {len(test_data)} validation samples")
            else:
                test_data = x_validation.iloc[:50]
                print(f"  Computing SHAP for first 50 validation samples")
            
            test_tensor = torch.FloatTensor(test_data.values)
            
            # Create DeepExplainer
            print(f"  Creating DeepExplainer...")
            
            # DeepExplainer expects a function that takes tensors and returns tensors
            # We need to wrap the RNN to return only the specific gene's output
            class GeneSpecificRNNWrapper(torch.nn.Module):
                def __init__(self, model, gene_idx):
                    super().__init__()
                    self.model = model
                    self.gene_idx = gene_idx
                
                def forward(self, x):
                    # Get full predictions
                    full_predictions, _ = self.model(x)
                    # Return only the specific gene's predictions
                    return full_predictions[:, self.gene_idx:self.gene_idx+1]
            
            wrapped_model = GeneSpecificRNNWrapper(rnn_model, gene_idx)
            wrapped_model.eval()
            
            # Create explainer
            rnn_explainer = shap.DeepExplainer(wrapped_model, background_tensor)
            
            # Compute SHAP values
            print(f"  Computing SHAP values (this should be fast!)...")
            rnn_shap_values = rnn_explainer.shap_values(test_tensor)
            
            # DeepExplainer returns numpy array
            if isinstance(rnn_shap_values, list):
                rnn_shap_values = rnn_shap_values[0]
            
            # Ensure 2D shape (samples, features)
            if rnn_shap_values.ndim == 3:
                rnn_shap_values = rnn_shap_values.squeeze(-1)
            
            # Get expected value (base prediction on background)
            with torch.no_grad():
                background_pred, _ = rnn_model(background_tensor)
                rnn_expected_value = float(background_pred[:, gene_idx].mean().item())
            
            shap_results['RNN'][gene_name] = {
                'shap_values': rnn_shap_values,
                'expected_value': rnn_expected_value,
                'explainer_type': 'DeepExplainer',
                'n_samples_computed': len(test_data)
            }
            
            print(f"  ✓ RNN SHAP shape: {rnn_shap_values.shape}")
            print(f"  ✓ Expected value: {rnn_expected_value:.4f}")
            print(f"  ✓ Mean |SHAP|: {np.abs(rnn_shap_values).mean():.6f}")
            
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

# Save SHAP values for each model
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
    'models_computed': [m for m in ['MLR', 'XGBRF', 'RNN'] 
                        if any(shap_results[m].get(g) is not None for g in GENES_OF_INTEREST)],
    'explainer_types': {
        'MLR': 'LinearExplainer',
        'XGBRF': 'TreeExplainer',
        'RNN': 'DeepExplainer' if ENABLE_RNN else 'Not computed'
    },
    'rnn_background_samples': BACKGROUND_SAMPLES_RNN if ENABLE_RNN else None,
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
if not ENABLE_RNN:
    print("2. Set ENABLE_RNN=True to compute RNN SHAP values (~2-3 minutes)")
print("3. Create publication-quality waterfall/beeswarm plots")
print("4. Compare feature importance across models")
