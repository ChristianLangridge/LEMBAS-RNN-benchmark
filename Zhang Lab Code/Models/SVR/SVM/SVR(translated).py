import pandas as pd
import cupy as cp
from cuml.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np

# -------------------------------
# 1. Load data and convert to GPU format
# -------------------------------
X_df = pd.read_csv('/proj/berzelius-2025-61/users/x_kejli/svm/TF.tsv',
                   sep='\t', index_col=0).astype(np.float32)
y_df = pd.read_csv('/proj/berzelius-2025-61/users/x_kejli/svm/Geneexpression.tsv',
                   sep='\t', index_col=0).astype(np.float32)

# Convert to cupy array
X = cp.asarray(X_df.values)
y = cp.asarray(y_df.values)

# -------------------------------
# 2. Use numpy indexing to split on GPU
# -------------------------------
# First generate indices for all samples
indices = np.arange(X.shape[0])
# Split indices on CPU
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42)

# Use numpy indexing directly on cupy
X_train = X[train_idx]
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

# Retain raw test set row indices
test_index = y_df.index[test_idx]

# -------------------------------
# 3. Manual multi-output regression
# -------------------------------
print("Begin training multi-output SVR model...")
models = []
n_targets = y_train.shape[1]
for col in range(n_targets):
    print(f"  Training target {col+1}/{n_targets}...")
    svr = SVR(kernel='rbf', C=1.0, gamma='scale')
    svr.fit(X_train, y_train[:, col])
    models.append(svr)
print("Training complete!")

# -------------------------------
# 4. Predict all target variables
# -------------------------------
y_pred = cp.zeros_like(y_test)
for i, m in enumerate(models):
    y_pred[:, i] = m.predict(X_test)

# -------------------------------
# 5. Save predictions (keep test set indices)
# -------------------------------
y_pred_df = pd.DataFrame(
    y_pred.get(),
    columns=y_df.columns,
    index=test_index
)
y_pred_df.to_csv(
    '/proj/berzelius-2025-61/users/x_kejli/svm/Y_hat_SVR.tsv',
    sep='\t'
)

# -------------------------------
# 6. Metric calculation (all on GPU)
# -------------------------------
def gpu_pearson(y_true, y_pred):
    mu_t = cp.mean(y_true, axis=0)
    mu_p = cp.mean(y_pred, axis=0)
    cov  = cp.mean((y_true - mu_t) * (y_pred - mu_p), axis=0)
    return cov / (cp.std(y_true, axis=0) * cp.std(y_pred, axis=0))

mse = cp.mean((y_test - y_pred) ** 2).get()
r2  = (1 - cp.sum((y_test - y_pred) ** 2) /
       cp.sum((y_test - cp.mean(y_test, axis=0)) ** 2)).get()
pearson_mean = cp.mean(gpu_pearson(y_test, y_pred)).get()

print(f"\nTest set MSE: {mse:.6f}")
print(f"Test set R²: {r2:.6f}")
print(f"Mean Pearson correlation: {pearson_mean:.6f}")
print("All processes completed!")
