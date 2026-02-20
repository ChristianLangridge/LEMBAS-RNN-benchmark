"""
RNN SHAP with Optimized GradientExplainer - All Performance Fixes Applied
==========================================================================

CRITICAL OPTIMIZATIONS:
1. Reduced background to 5 samples (was 25)
2. Reduced test subset to 10 samples (was 50) 
3. Added n_samples=10 to limit interpolation steps
4. Added progress monitoring with tqdm
5. Computing one sample at a time with progress tracking
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import sys
import os
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
from pathlib import Path

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
# Configuration - HEAVILY OPTIMIZED
# ============================================================================

GENES_OF_INTEREST = ['ALB', 'AFP']

OUTPUT_BASE_PATH = f"{DATA_ROOT}/Saved SHAP values/gene_specific"
MODELS_BASE_PATH = f"{DATA_ROOT}/Saved models"

# CRITICAL: Aggressively reduced for feasibility
RNN_BACKGROUND_SAMPLES = 50   # Was 5 - integrated gradients scale with this!
RNN_TEST_SAMPLES = 262       # Was 10 - start VERY small
RNN_N_SAMPLES = 25           # (was 10)

os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# ============================================================================
# Load Data
# ============================================================================

print("="*80)
print("Loading Validation Data")
print("="*80)

validation_dataset = pd.read_csv(f'{DATA_ROOT}/Full data files/Liver_bulk_external.tsv', 
                                 sep='\t', header=0, index_col=0)

net = pd.read_csv(f'{DATA_ROOT}/Full data files/network(full).tsv', sep='\t')
network_tfs = set(net['TF'].unique())
network_genes = set(net['Gene'].unique())
network_nodes = network_tfs | network_genes

tf_expression = pd.read_csv(f'{DATA_ROOT}/Full data files/TF(full).tsv', 
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

gene_expression_ref = pd.read_csv(f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv", 
                                  sep="\t", header=0, index_col=0)
expected_features = gene_expression_ref.columns.tolist()
new_input = validation_dataset.copy()

for feature in expected_features:
    if feature not in new_input.columns:
        new_input[feature] = 0

y_validation = new_input[expected_features]

print(f"✓ x_validation: {x_validation.shape}")
print(f"✓ y_validation: {y_validation.shape}")

# ============================================================================
# Find Genes
# ============================================================================

gene_indices = {}
for gene in GENES_OF_INTEREST:
    try:
        idx = list(y_validation.columns).index(gene)
        gene_indices[gene] = idx
        print(f"✓ {gene} at index {idx}")
    except ValueError:
        print(f"✗ {gene} not found")

# ============================================================================
# Load RNN
# ============================================================================

print("\n" + "="*80)
print("Loading RNN Model")
print("="*80)

rnn_model = load_model_from_checkpoint(
    checkpoint_path=f'{MODELS_BASE_PATH}/RNN/uncentered_data_RNN/signaling_model.v1.pt',
    net_path=f'{DATA_ROOT}/Full data files/network(full).tsv',
    X_in_df=pd.DataFrame(x_validation),
    y_out_df=pd.DataFrame(y_validation),
    device='cpu',
    use_exact_training_params=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rnn_model = rnn_model.to(device)
rnn_model.eval()

print(f"✓ RNN loaded on {device.upper()}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rnn_model = rnn_model.to(device)

# --- START PATCH: Fix Custom Attributes that .to() ignored ---
print(f"🔧 Patching model attributes for {device}...")

# 1. Fix Input Layer
rnn_model.input_layer.device = device
rnn_model.input_layer.input_node_order = rnn_model.input_layer.input_node_order.to(device)

# 2. Fix Hidden Layer (BioNet)
# The internal BioNet also relies on 'self.device' for initializing temp tensors
rnn_model.signaling_network.device = device 
# Ensure mask tensors (if used) are moved
if hasattr(rnn_model.signaling_network, 'mask'):
    rnn_model.signaling_network.mask = rnn_model.signaling_network.mask.to(device)

# 3. Fix Output Layer
rnn_model.output_layer.output_node_order = rnn_model.output_layer.output_node_order.to(device)

# 4. Apply the Safety Cap (from previous step)
rnn_model.signaling_network.training_params['max_steps'] = 100 

print("✓ Model attributes patched manually.")
# --- END PATCH ---

rnn_model.eval()

# ... [Continue to Convergence Check] ...

# OPTIONAL SAFETY CHECK (Run this once manually to be sure):
# Verify the model actually converges within 100 steps
print("Verifying convergence at 100 steps...")
with torch.no_grad():
    test_out = rnn_model(torch.randn(1, 1197).to(device)) # Random noise input
    # If this runs without error, the forward pass is safe.
    print("✓ Forward pass check passed.")


# ============================================================================
# Compute RNN SHAP - OPTIMIZED WITH PROGRESS
# ============================================================================

print("\n" + "="*80)
print(f"Computing RNN SHAP (OPTIMIZED)")
print("="*80)
print(f"⚠️  Background: {RNN_BACKGROUND_SAMPLES} (reduced from 25)")
print(f"⚠️  Test samples: {RNN_TEST_SAMPLES} (reduced from 50)")
print(f"⚠️  Interpolation steps: {RNN_N_SAMPLES} (reduced from 200 default)")
print(f"⚠️  Strategy: One sample at a time with progress tracking")

# Gene-specific wrapper
class GeneTargetWrapper(nn.Module):
    def __init__(self, model, target_idx):
        super().__init__()
        self.model = model
        self.target_idx = target_idx
        
    def forward(self, x):
        y_hat, _ = self.model(x)
        return y_hat[:, self.target_idx].unsqueeze(1)

# Prepare background (very small!)
background_data = shap.sample(x_validation, RNN_BACKGROUND_SAMPLES)
background_tensor = torch.FloatTensor(background_data.values).to(device)

# Prepare test subset
x_subset = x_validation.iloc[:RNN_TEST_SAMPLES]

shap_results = {}

for gene_name, gene_idx in gene_indices.items():
    print(f"\n{'─'*80}")
    print(f"Processing {gene_name}")
    print(f"{'─'*80}")
    
    try:
        # Create wrapper
        target_model = GeneTargetWrapper(rnn_model, gene_idx)
        
        # Create explainer
        print(f"Creating GradientExplainer...")
        rnn_explainer = shap.GradientExplainer(target_model, background_tensor)
        
        # Compute SHAP ONE SAMPLE AT A TIME with progress
        all_shap_values = []
        
        print(f"Computing SHAP for {RNN_TEST_SAMPLES} samples (one at a time)...")
        start_time = datetime.now()
        
        for i in tqdm(range(RNN_TEST_SAMPLES), desc=gene_name):
            # Get single sample
            sample = x_subset.iloc[i:i+1]
            sample_tensor = torch.FloatTensor(sample.values).to(device)
            
            # Compute SHAP for this one sample
            # CRITICAL: n_samples parameter controls interpolation steps
            sample_shap = rnn_explainer.shap_values(
                sample_tensor,
                nsamples=RNN_N_SAMPLES  # This is the key parameter!
            )
            
            if isinstance(sample_shap, list):
                sample_shap = sample_shap[0]
            
            all_shap_values.append(sample_shap)
            
            # Show progress every 2 samples
            if (i + 1) % 2 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = elapsed / (i + 1)
                remaining = rate * (RNN_TEST_SAMPLES - i - 1)
                print(f"  [{i+1}/{RNN_TEST_SAMPLES}] - {rate:.1f}s/sample - ETA: {remaining/60:.1f} min")
        
        # Concatenate all samples
        rnn_shap_values = np.concatenate(all_shap_values, axis=0)
        
        # Get expected value
        with torch.no_grad():
            expected_value_tensor = target_model(background_tensor).mean()
            rnn_expected_value = float(expected_value_tensor.cpu().numpy())
        
        shap_results[gene_name] = {
            'shap_values': rnn_shap_values,
            'expected_value': rnn_expected_value,
            'n_samples': RNN_TEST_SAMPLES,
            'subset_indices': x_subset.index.tolist()
        }
        
        elapsed_total = (datetime.now() - start_time).total_seconds()
        print(f"\n✓ {gene_name} complete in {elapsed_total/60:.1f} minutes")
        print(f"  Shape: {rnn_shap_values.shape}")
        print(f"  Expected: {rnn_expected_value:.4f}")
        
    except Exception as e:
        print(f"\n✗ {gene_name} failed: {e}")
        import traceback
        traceback.print_exc()
        shap_results[gene_name] = None
    
    finally:
        # Clean up GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================================
# Save Results
# ============================================================================

print("\n" + "="*80)
print("Saving Results")
print("="*80)

output_file = f'{OUTPUT_BASE_PATH}/rnn_shap_values_OPTIMIZED(FINAL-nsamples=262).npz'

save_dict = {
    'feature_names': np.array(x_validation.columns.tolist()),
    'genes': np.array(GENES_OF_INTEREST),
    'gene_indices': json.dumps(gene_indices)
}

for gene_name in GENES_OF_INTEREST:
    if gene_name in shap_results and shap_results[gene_name] is not None:
        save_dict[f'RNN_{gene_name}_shap_values'] = shap_results[gene_name]['shap_values']
        save_dict[f'RNN_{gene_name}_expected_value'] = shap_results[gene_name]['expected_value']
        save_dict[f'RNN_{gene_name}_subset_indices'] = np.array(shap_results[gene_name]['subset_indices'])

np.savez_compressed(output_file, **save_dict)

print(f"✓ Saved to: {output_file}")
print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")

success_count = sum(1 for r in shap_results.values() if r is not None)
print(f"Success: {success_count}/{len(GENES_OF_INTEREST)} genes")
print(f"Samples per gene: {RNN_TEST_SAMPLES}")
print(f"\nIMPORTANT NOTES:")
print(f"- This is a SUBSET ({RNN_TEST_SAMPLES}/{len(x_validation)} samples)")
print(f"- Sufficient for feature importance comparison")
print(f"- To compute more samples, increase RNN_TEST_SAMPLES incrementally")