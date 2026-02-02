import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append('/home/christianl/Zhang-Lab/Zhang Lab Code/Tuning/uncentered_RNN_tuning')
from RNN_reconstructor import load_model_from_checkpoint

print("="*80)
print("LOADING DATA")
print("="*80)

# ============================================================================
# Load training/test data
# ============================================================================
print("\nLoading training/test data...")
gene_expression = pd.read_csv('~/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv', 
                              sep='\t', header=0, index_col=0)
tf_expression = pd.read_csv('~/Zhang-Lab/Zhang Lab Data/Full data files/TF(full).tsv', 
                            sep='\t', header=0, index_col=0)

# Filter TFs
net = pd.read_csv('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/network(full).tsv', sep='\t')
network_tfs = set(net['TF'].unique())
network_genes = set(net['Gene'].unique())
network_nodes = network_tfs | network_genes
usable_features = [tf for tf in tf_expression.columns if tf in network_nodes]

x = tf_expression[usable_features]
y = gene_expression

# Train/test split (same seed as your training)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=888)

print(f"✓ Training data: x_train {x_train.shape}, y_train {y_train.shape}")
print(f"✓ Test data: x_test {x_test.shape}, y_test {y_test.shape}")

# ============================================================================
# Load and process external validation data
# ============================================================================
"Kejun's 05-Addgene.py script: Add genes in external validation gene expression data to make its dimension matches train data"

# Load input data
input_file = '/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv' #output from 05 geneexpression
new_input = pd.read_csv('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/Liver_bulk_external.tsv', sep="\t", index_col=0)

# Read the expected features from the first line of the reference file
with open("/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv", "r", encoding="utf-8") as file:
    header_line = file.readline().strip()  # Read the first line and strip whitespace

# Split the header line into feature names
expected_features = header_line.split("\t")

# Identify missing features in the input data
missing_features = [feature for feature in expected_features if feature not in new_input.columns]

# Fill missing features with zeros
for feature in missing_features:
    new_input[feature] = 0

# Reorder columns to match the expected feature order
new_input = new_input[expected_features]

# Save the updated dataframe to a new file
output_file = "Liver_bulk_external(PROCESSED).tsv"
new_input.to_csv(output_file, sep="\t")

y_val = 

# ============================================================================
# MLR Model
# ============================================================================
print("\n[1/3] Loading MLR model...")
mlr_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/MLR/MLR_v3/MLR_model_v4(uncentered[FINAL]).joblib'
mlr_loaded = joblib.load(mlr_model_path)

mlr_y_pred_train = mlr_loaded.predict(x_train)
mlr_y_pred_test = mlr_loaded.predict(x_test)
mlr_y_pred_external = mlr_loaded.predict(x_val)

print(f"✓ MLR training predictions: {mlr_y_pred_train.shape}")
print(f"✓ MLR test predictions: {mlr_y_pred_test.shape}")
print(f"✓ MLR external predictions: {mlr_y_pred_external.shape}")

# ============================================================================
# XGBRF Model
# ============================================================================
print("\n[2/3] Loading XGBRF models...")
xgbrf_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/XGBRF/XGBRF_v5/all_models_batch_XGBRF[uncentered_REALFINAL].joblib'
xgbrf_loaded = joblib.load(xgbrf_model_path)

xgbrf_y_pred_train = np.column_stack([model.predict(x_train) for model in xgbrf_loaded])
xgbrf_y_pred_test = np.column_stack([model.predict(x_test) for model in xgbrf_loaded])
xgbrf_y_pred_external = np.column_stack([model.predict(external_x) for model in xgbrf_loaded])

print(f"✓ XGBRF training predictions: {xgbrf_y_pred_train.shape}")
print(f"✓ XGBRF test predictions: {xgbrf_y_pred_test.shape}")
print(f"✓ XGBRF external predictions: {xgbrf_y_pred_external.shape}")

# ============================================================================
# RNN Model
# ============================================================================
print("\n[3/3] Loading RNN model...")

# Training predictions
print("  Generating training predictions...")
RNN_train = load_model_from_checkpoint(
    checkpoint_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/RNN/uncentered_data_RNN/signaling_model.v1.pt',
    net_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/network(full).tsv',
    X_in_df=pd.DataFrame(x_train),
    y_out_df=pd.DataFrame(y_train),
    device='cpu',
    use_exact_training_params=True)

with torch.no_grad():
    rnn_y_pred_train, _ = RNN_train(RNN_train.X_in)
    rnn_y_pred_train = rnn_y_pred_train.detach().numpy()

# Test predictions
print("  Generating test predictions...")
RNN_test = load_model_from_checkpoint(
    checkpoint_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/RNN/uncentered_data_RNN/signaling_model.v1.pt',
    net_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/network(full).tsv',
    X_in_df=pd.DataFrame(x_test),
    y_out_df=pd.DataFrame(y_test),
    device='cpu',
    use_exact_training_params=True)

with torch.no_grad():
    rnn_y_pred_test, _ = RNN_test(RNN_test.X_in)
    rnn_y_pred_test = rnn_y_pred_test.detach().numpy()

# External predictions
print("  Generating external predictions...")
RNN_external = load_model_from_checkpoint(
    checkpoint_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/RNN/uncentered_data_RNN/signaling_model.v1.pt',
    net_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/network(full).tsv',
    X_in_df=pd.DataFrame(external_x),
    y_out_df=pd.DataFrame(external_y),
    device='cpu',
    use_exact_training_params=True)

with torch.no_grad():
    rnn_y_pred_external, _ = RNN_external(RNN_external.X_in)
    rnn_y_pred_external = rnn_y_pred_external.detach().numpy()

print(f"✓ RNN training predictions: {rnn_y_pred_train.shape}")
print(f"✓ RNN test predictions: {rnn_y_pred_test.shape}")
print(f"✓ RNN external predictions: {rnn_y_pred_external.shape}")

# ============================================================================
# Save everything
# ============================================================================
print("\n" + "="*80)
print("SAVING PREDICTIONS")
print("="*80)

output_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved predictions/model_predictions_uncentered_v2_with_external.npz'

print(f"\nSaving predictions to: {output_path}")
np.savez_compressed(
    output_path,
    # Training data
    y_train=y_train.values,
    y_train_columns=y_train.columns.values,
    mlr_y_pred_train=mlr_y_pred_train,
    xgbrf_y_pred_train=xgbrf_y_pred_train,
    rnn_y_pred_train=rnn_y_pred_train,
    # Test data
    y_test=y_test.values,
    y_test_columns=y_test.columns.values,
    mlr_y_pred_test=mlr_y_pred_test,
    xgbrf_y_pred_test=xgbrf_y_pred_test,
    rnn_y_pred_test=rnn_y_pred_test,
    # External validation data
    y_external=external_y.values,
    y_external_columns=external_y.columns.values,
    external_sample_ids=external_y.index.values,
    mlr_y_pred_external=mlr_y_pred_external,
    xgbrf_y_pred_external=xgbrf_y_pred_external,
    rnn_y_pred_external=rnn_y_pred_external
)

print("\n✓ Predictions saved successfully!")
print(f"✓ File size: {os.path.getsize(output_path) / 1e6:.2f} MB")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nDatasets saved:")
print(f"  • Training:   {y_train.shape[0]} samples × {y_train.shape[1]} genes")
print(f"  • Test:       {y_test.shape[0]} samples × {y_test.shape[1]} genes")
print(f"  • External:   {external_y.shape[0]} samples × {external_y.shape[1]} genes")
print(f"\nModels saved: MLR, XGBRFRegressor, RNN")
print(f"Predictions per model: training, test, external")
print("\n" + "="*80)