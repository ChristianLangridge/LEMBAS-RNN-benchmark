import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration and utilities
# ============================================================================

MODEL_COLORS = {
    'RNN': '#1f77b4',           # Deep blue
    'XGBRFRegressor': '#ff7f0e', # Orange
    'MLR': '#2ca02c'            # Green
}

DPI = 300
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_TRIPLE = (15, 5)
FIGSIZE_WIDE = (14, 8)

def set_publication_style():
    """Apply consistent publication-quality styling."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    })

# ============================================================================
# Load pre-computed predictions
# ============================================================================

print("Loading pre-computed predictions...")
data_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved predictions/model_predictions_uncentered_v1.npz'
data = np.load(data_path, allow_pickle=True)

data_path_external = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved predictions/model_predictions_uncentered_v1.npz'
data_external = np.load(data_path_external, allow_pickle=True)

# Reconstruct training data
y_train = pd.DataFrame(
    data['y_train'],
    columns=data['y_train_columns']
)

predictions_train = {
    "MLR": data['mlr_y_pred_train'],
    "XGBRFRegressor": data['xgbrf_y_pred_train'],
    "RNN": data['rnn_y_pred_train']
}

# Reconstruct test data
y_test = pd.DataFrame(
    data['y_test'],
    columns=data['y_test_columns']
)

predictions_test = {
    "MLR": data['mlr_y_pred_test'],
    "XGBRFRegressor": data['xgbrf_y_pred_test'],
    "RNN": data['rnn_y_pred_test']
}

y_validation = pd.DataFrame(
    data_external['y_validation'],
    columns=data_external['y_validation_columns']
)

predictions_validation = {
    "MLR": data_external['mlr_y_pred_val'],
    "XGBRFRegressor": data_external['xgbrf_y_pred_val'],
    "RNN": data_external['rnn_y_pred_val']
}




print(f"✓ Loaded predictions for {len(predictions_train)} models")
print(f"  Training samples: {y_train.shape[0]}, Genes: {y_train.shape[1]}")
print(f"  Test samples: {y_test.shape[0]}, Genes: {y_test.shape[1]}")
print(f"  Val samples: {y_validation.shape[0]}, Genes: {y_validation.shape[1]}")

# ============================================================================
# Metrics functions
# ============================================================================

def compute_metrics(y_true, y_pred):
    """Compute metrics on flattened data."""
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    pearson_r, p_value = pearsonr(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    return {
        'r2': r2,
        'pearson_r': pearson_r,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae
    }

def compute_metrics_per_gene(y_true_df, y_pred_array):
    """Compute per-gene metrics."""
    results = []
    
    for i, gene_name in enumerate(y_true_df.columns):
        y_t = y_true_df.iloc[:, i].values
        y_p = y_pred_array[:, i]
        
        if np.var(y_t) > 1e-10:
            pearson_r, p_value = pearsonr(y_t, y_p)
            r2 = r2_score(y_t, y_p)
            
            results.append({
                'gene': gene_name,
                'gene_idx': i,
                'r2': r2,
                'pearson_r': pearson_r,
                'p_value': p_value
            })
    
    return pd.DataFrame(results)

def quick_r2_comparison(y_true, y_pred, model_name="Model"):
    """Quick comparison of flattened vs median per-gene R²."""
    # Flattened R²
    r2_flattened = r2_score(y_true.ravel(), y_pred.ravel())
    
    # Per-gene R²
    n_genes = y_true.shape[1]
    r2_per_gene = []
    variances = []
    
    for gene_idx in range(n_genes):
        y_t = y_true[:, gene_idx]
        y_p = y_pred[:, gene_idx]
        variance = np.var(y_t)
        
        if variance > 1e-10:
            r2 = r2_score(y_t, y_p)
            r2_per_gene.append(r2)
            variances.append(variance)
    
    r2_per_gene = np.array(r2_per_gene)
    variances = np.array(variances)
    
    # Median R²
    r2_median = np.median(r2_per_gene)
    
    # Variance-weighted R²
    r2_weighted = np.sum(r2_per_gene * variances) / np.sum(variances)
    
    # Correlation between variance and R²
    corr = np.corrcoef(variances, r2_per_gene)[0, 1]
    
    print(f"\n{'='*60}")
    print(f"{model_name} - R² Comparison")
    print(f"{'='*60}")
    print(f"Flattened R² (pooled):           {r2_flattened:.6f}")
    print(f"Median R² (per-gene):            {r2_median:.6f}")
    print(f"Weighted R² (sklearn .score()):  {r2_weighted:.6f}")
    print(f"\nDifferences:")
    print(f"Flattened - Median:              {r2_flattened - r2_median:+.6f}")
    print(f"Weighted - Median:               {r2_weighted - r2_median:+.6f}")
    print(f"\nVariance-R² correlation:         {corr:+.4f}")
    if corr > 0.1:
        print(f"→ Model performs BETTER on high-variance genes")
    elif corr < -0.1:
        print(f"→ Model performs BETTER on low-variance genes")
    else:
        print(f"→ Model performance is independent of gene variance")
    print(f"{'='*60}\n")
    
    return {
        'flattened': r2_flattened,
        'median': r2_median,
        'weighted': r2_weighted,
        'corr': corr
    }

# ============================================================================
# Ready for analysis!
# ============================================================================

print("\n✓ All functions loaded. Ready for analysis!")