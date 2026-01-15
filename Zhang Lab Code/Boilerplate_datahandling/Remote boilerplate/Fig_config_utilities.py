import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import joblib 
import glob
import os
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import torch

import sys
sys.path.append('/home/christianl/Zhang-Lab/Zhang Lab Code/Tuning/uncentered_RNN_tuning')
from RNN_reconstructor import load_model_from_checkpoint

# configuration and utilities 
# publication-style figure settings

MODEL_COLORS = {
    'RNN': '#1f77b4',           # Deep blue
    'XGBRFRegressor': '#ff7f0e', # Orange
    'Linear': '#2ca02c'          # Green
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

#####################################################################################

##### getting y_test and each model's y_pred  

##### loading and centering data

# reading in full data files
gene_expression = pd.read_csv(('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv'), sep='\t', header=0, index_col=0)
tf_expression = pd.read_csv(('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/TF(full).tsv'), sep='\t', header=0, index_col=0)

# Removing the 'TF' filler column I identified causing me data mismatches when loading all models up with the network.tsv 15/01/26
tf_expression = tf_expression.drop(columns=['TF']) 

# Split into training, testing and validation sets and into numpy arrays + combining dataframes
x = tf_expression
y = gene_expression
# First split: 70% train and 30% temp (test + val)
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=888)

# Second split: split the temp set into 20% test and 10% val (which is 2/3 and 1/3 of temp)
x_test, x_val, y_test, y_val = train_test_split(
    x_temp, y_temp, test_size=1/3, random_state=888)

# getting gene IDs into a single vector for future analysis
y_train_gene_names = list(y_train.columns)

###############################################################################################

#### centering script 
# column-wise centering for training set (each gene is a column, each row is an instance)
#x_train_col_means = x_train.mean(axis=0)
#x_train_centered = x_train - x_train_col_means

#y_train_col_means = y_train.mean(axis=0)
#y_train_centered = y_train - y_train_col_means

# for test set 
#x_test_centered = x_test - x_train_col_means
#y_test_centered = y_test - y_train_col_means

# for val set
#x_val_centered = x_val - x_train_col_means
#y_val_centered = y_val - y_train_col_means

###############################################################################################

##### loading MLR model (v3, trained on uncentered data)
mlr_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/MLR/MLR_v3/MLR_model_v4(uncentered[FINAL]).joblib'
mlr_loaded = joblib.load(mlr_model_path)
mlr_y_pred = mlr_loaded.predict(x_test)          
print(type(mlr_y_pred), mlr_y_pred.shape)

##### loading XGBRF models (v1)
#xgbrf_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/Random Forest/Saved_Models_XGBRF_v1.pkl'
# find all saved models, compute xgbrf_y_pred (trained on uncentered data unlike MLR
# so need to keep it as x_test to avoid destroying performance) --> ultimately decided 
# to universally use centered data and just retrain XGBRFRegressor() on x_test_centered (19/12/25)
#with open(xgbrf_model_path, 'rb') as f:
#    models = pickle.load(f)
#xgbrf_y_pred = np.column_stack([model.predict(x_test_centered) for model in models])

##### loading XGBRF models (v4, trained on uncentered data)
xgbrf_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/XGBRF/XGBRF_v4/all_models_batch_XGBRF[uncentered_FINAL].joblib'
xgbrf_loaded = joblib.load(xgbrf_model_path)
xgbrf_y_pred = np.column_stack([model.predict(x_test) for model in xgbrf_loaded])  
print(type(xgbrf_y_pred), xgbrf_y_pred.shape)

##### loading RNN (v1, trained on uncentered data)
RNN_loaded = load_model_from_checkpoint(
                checkpoint_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/RNN/uncentered_data_RNN/signaling_model.v1.pt',
                net_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/network(full).tsv',
                X_in_df=pd.DataFrame(x_test),  # passing as df not tensors
                y_out_df=pd.DataFrame(y_test),  # passing as df not tensors
                device='cpu',
                use_exact_training_params=True)

# convert x_test to tensor and pass through model
with torch.no_grad():  # Disable gradients for inference
    rnn_y_pred, _ = RNN_loaded(torch.tensor(x_test.values, dtype=torch.float32))
    rnn_y_pred = rnn_y_pred.detach().numpy()
print(type(rnn_y_pred), rnn_y_pred.shape)

####################################################################################################

##### assemble all y_pred into dictionary, where values are 2D numpy arrays

predictions = {
    "MLR": mlr_y_pred,
    "XGBRFRegressor": xgbrf_y_pred,
    "RNN": rnn_y_pred
}

####################################################################################################

# performance metrics, globally, for further analysis  
# flattens 2D numpy arrays into 1D arrays where each gene is treated equally

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
    
# same as computer_metrics, but at per-gene resolution
def compute_metrics_per_gene(y_true, y_pred):
    """Compute metrics for each gene separately (recommended for multi-output)."""   
    n_genes = y_true.shape[1]
    results = []
    
    for gene_idx in range(n_genes):
        y_t = y_true[:, gene_idx]
        y_p = y_pred[:, gene_idx]
        
        # Skip genes with no variance
        if np.var(y_t) > 1e-10:
            pearson_r, p_value = pearsonr(y_t, y_p)
            r2 = r2_score(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            mae = mean_absolute_error(y_t, y_p)
            
            results.append({
                'gene_idx': gene_idx,
                'r2': r2,
                'pearson_r': pearson_r,
                'p_value': p_value,
                'rmse': rmse,
                'mae': mae
            })
    
    return pd.DataFrame(results)


####################################################################################################

# using median r2 as a way of tracking model performance in a way more robust to specific
# genes with outlying variance (ie. odd predictions)
# median R² vs flattened R²
def quick_r2_comparison(y_true, y_pred, model_name="Model"):
    """
    Quick comparison of flattened vs median per-gene R².
    """
    # flattened R²
    r2_flattened = r2_score(y_true.ravel(), y_pred.ravel())
    
    # per-gene R²
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
    
    # median R²
    r2_median = np.median(r2_per_gene)
    
    # variance-weighted R² (score.() behaviour)
    r2_weighted = np.sum(r2_per_gene * variances) / np.sum(variances)
    
    # correlation between variance and R²
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

####################################################################################################