import numpy as np
import pandas as pd
import torch
import joblib
import sys
import os

sys.path.append('/home/christianl/Zhang-Lab/Zhang Lab Code/Tuning/uncentered_RNN_tuning')
from RNN_reconstructor import load_model_from_checkpoint

print("="*70)
print("Generating x_val, y_val and model predictions")
print("="*70)

# ============================================================================
# PART 1: Generate x_validation (TF features)
# ============================================================================

print("\n[1/6] Loading external liver bulk validation data...")
validation_dataset = pd.read_csv('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/Liver_bulk_external.tsv', 
                                 sep='\t', header=0, index_col=0)

print(f"   External data shape: {validation_dataset.shape}")

print("\n[2/6] Generating x_validation (TF features)...")
# Load network to identify usable TF features
net = pd.read_csv('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/network(full).tsv', sep='\t')
network_tfs = set(net['TF'].unique())
network_genes = set(net['Gene'].unique())
network_nodes = network_tfs | network_genes

# Load reference TF expression data
tf_expression = pd.read_csv('~/Zhang-Lab/Zhang Lab Data/Full data files/TF(full).tsv', 
                            sep='\t', header=0, index_col=0)

# Determine usable features (TFs that are in the network)
usable_features = [tf for tf in tf_expression.columns if tf in network_nodes]

# Identify which usable features are present in validation data
present_features = [tf for tf in usable_features if tf in validation_dataset.columns]
missing_features = [tf for tf in usable_features if tf not in validation_dataset.columns]

print(f"   Usable TF features: {len(usable_features)}")
print(f"   Present in validation data: {len(present_features)}")
print(f"   Missing (filled with 0): {len(missing_features)}")

# Create x_validation
x_validation = pd.DataFrame(index=validation_dataset.index)

# Add present features
for feature in present_features:
    x_validation[feature] = validation_dataset[feature]

# Add missing features with zeros
for feature in missing_features:
    x_validation[feature] = 0

# Reorder columns to match the expected feature order
x_validation = x_validation[usable_features]

print(f"   x_validation shape: {x_validation.shape}")

# ============================================================================
# PART 2: Generate y_validation (Gene expression features)
# Using Kejun's gene match up script
# ============================================================================

print("\n[3/6] Generating y_validation (Gene expression features)...")

# Load input data
new_input = validation_dataset

# Load the reference gene expression file
gene_expression_ref = pd.read_csv("/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv", 
                                  sep="\t", header=0, index_col=0)

# Get expected features from columns
expected_features = gene_expression_ref.columns.tolist()

# Identify missing features in the input data
missing_features_y = [feature for feature in expected_features if feature not in new_input.columns]

print(f"   Expected gene features: {len(expected_features)}")
print(f"   Missing (filled with 0): {len(missing_features_y)}")

# Fill missing features with zeros
for feature in missing_features_y:
    new_input[feature] = 0

# Reorder columns to match the expected feature order
new_input = new_input[expected_features]

y_validation = new_input

print(f"   y_validation shape: {y_validation.shape}")

# ============================================================================
# PART 3: Load models and generate predictions on y_validation
# ============================================================================

print("\n[4/6] Loading MLR model and generating predictions...")
mlr_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/MLR/MLR_v3/MLR_model_v4(uncentered[FINAL]).joblib'
mlr_loaded = joblib.load(mlr_model_path)
mlr_y_pred_val = mlr_loaded.predict(x_validation)
print(f"   MLR predictions shape: {mlr_y_pred_val.shape}")

print("\n[5/6] Loading XGBRF models and generating predictions...")
xgbrf_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/XGBRF/XGBRF_v5/all_models_batch_XGBRF[uncentered_REALFINAL].joblib'
xgbrf_loaded = joblib.load(xgbrf_model_path)
xgbrf_y_pred_val = np.column_stack([model.predict(x_validation) for model in xgbrf_loaded])
print(f"   XGBRF predictions shape: {xgbrf_y_pred_val.shape}")

print("\n[6/6] Loading RNN model and generating predictions...")
RNN_val = load_model_from_checkpoint(
    checkpoint_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/RNN/uncentered_data_RNN/signaling_model.v1.pt',
    net_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/network(full).tsv',
    X_in_df=pd.DataFrame(x_validation),
    y_out_df=pd.DataFrame(y_validation),
    device='cpu',
    use_exact_training_params=True)

with torch.no_grad():
    rnn_y_pred_val, _ = RNN_val(RNN_val.X_in)
    rnn_y_pred_val = rnn_y_pred_val.detach().numpy()

print(f"   RNN predictions shape: {rnn_y_pred_val.shape}")

# ============================================================================
# Save y_validation and predictions to npz file
# ============================================================================

output_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved predictions/model_predictions_validation_v1.npz'

print(f"\n{'='*70}")
print("Saving y_validation and model predictions...")
print(f"{'='*70}")
print(f"Output path: {output_path}")

np.savez_compressed(
    output_path,
    # Validation true outputs
    y_validation=y_validation.values,
    y_validation_columns=y_validation.columns.values,
    # Model predictions
    mlr_y_pred_val=mlr_y_pred_val,
    xgbrf_y_pred_val=xgbrf_y_pred_val,
    rnn_y_pred_val=rnn_y_pred_val
)

print("\n✓ Validation predictions saved successfully!")
print(f"   File size: {os.path.getsize(output_path) / 1e6:.2f} MB")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Validation samples:        {x_validation.shape[0]}")
print(f"Input features (TFs):      {x_validation.shape[1]}")
print(f"Output features (Genes):   {y_validation.shape[1]}")
print(f"\nPredictions saved for:")
print(f"  - MLR:            {mlr_y_pred_val.shape}")
print(f"  - XGBRFRegressor: {xgbrf_y_pred_val.shape}")
print(f"  - RNN:            {rnn_y_pred_val.shape}")
print(f"\nTo load this data:")
print(f"  data = np.load('{output_path}', allow_pickle=True)")
print(f"  y_val = pd.DataFrame(data['y_validation'], columns=data['y_validation_columns'])")
print(f"  mlr_pred = data['mlr_y_pred_val']")
print(f"  xgbrf_pred = data['xgbrf_y_pred_val']")
print(f"  rnn_pred = data['rnn_y_pred_val']")
print(f"{'='*70}")

print("\n✓ Script completed successfully!")