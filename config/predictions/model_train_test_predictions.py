import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
import sys
import os
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

print("Loading data...")
# Load data
gene_expression = pd.read_csv(f"{DATA_ROOT}/Full data files/Geneexpression (full).tsv", 
                              sep='\t', header=0, index_col=0)
tf_expression = pd.read_csv(f"{DATA_ROOT}/Full data files/TF(full).tsv", 
                            sep='\t', header=0, index_col=0)

# Filter TFs
net = pd.read_csv(f"{DATA_ROOT}/Full data files/network(full).tsv", sep='\t')
network_tfs = set(net['TF'].unique())
network_genes = set(net['Gene'].unique())
network_nodes = network_tfs | network_genes
usable_features = [tf for tf in tf_expression.columns if tf in network_nodes]

x = tf_expression[usable_features]
y = gene_expression

# Train/test split (same seed as your training)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=888)

print(f"Data shapes: x_train {x_train.shape}, y_train {y_train.shape}")

# ============================================================================
# Load models and generate predictions
# ============================================================================

print("\nLoading MLR model...")
mlr_model_path = f"{DATA_ROOT}/Saved models/MLR/MLR_v3/MLR_model_v4(uncentered[FINAL]).joblib"
mlr_loaded = joblib.load(mlr_model_path)
mlr_y_pred_train = mlr_loaded.predict(x_train)
mlr_y_pred_test = mlr_loaded.predict(x_test)
print(f"MLR predictions: {mlr_y_pred_train.shape}")

print("\nLoading XGBRF models...")
xgbrf_model_path = f"{DATA_ROOT}/Saved models/XGBRF/XGBRF_v5/all_models_batch_XGBRF[uncentered_REALFINAL].joblib"
xgbrf_loaded = joblib.load(xgbrf_model_path)
xgbrf_y_pred_train = np.column_stack([model.predict(x_train) for model in xgbrf_loaded])
xgbrf_y_pred_test = np.column_stack([model.predict(x_test) for model in xgbrf_loaded])
print(f"XGBRF predictions: {xgbrf_y_pred_train.shape}")

print("\nLoading RNN model...")
RNN_loaded = load_model_from_checkpoint(
    checkpoint_path=f"{DATA_ROOT}/Saved models/RNN/uncentered_data_RNN/signaling_model.v1.pt",
    net_path=f"{DATA_ROOT}/Full data files/network(full).tsv",
    X_in_df=pd.DataFrame(x_train),
    y_out_df=pd.DataFrame(y_train),
    device='cpu',
    use_exact_training_params=True)

with torch.no_grad():
    rnn_y_pred_train, _ = RNN_loaded(RNN_loaded.X_in)
    rnn_y_pred_train = rnn_y_pred_train.detach().numpy()

# Generate test predictions
RNN_test = load_model_from_checkpoint(
    checkpoint_path=f"{DATA_ROOT}/Saved models/RNN/uncentered_data_RNN/signaling_model.v1.pt",
    net_path=f"{DATA_ROOT}/Full data files/network(full).tsv",
    X_in_df=pd.DataFrame(x_test),
    y_out_df=pd.DataFrame(y_test),
    device='cpu',
    use_exact_training_params=True)

with torch.no_grad():
    rnn_y_pred_test, _ = RNN_test(RNN_test.X_in)
    rnn_y_pred_test = rnn_y_pred_test.detach().numpy()

print(f"RNN predictions: {rnn_y_pred_train.shape}")

# ============================================================================
# Save everything
# ============================================================================

output_path = f"{DATA_ROOT}/Saved predictions/model_predictions_uncentered_v1.npz"

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
    rnn_y_pred_test=rnn_y_pred_test
)

print("\n✓ Predictions saved successfully!")
print(f"File size: {os.path.getsize(output_path) / 1e6:.2f} MB")