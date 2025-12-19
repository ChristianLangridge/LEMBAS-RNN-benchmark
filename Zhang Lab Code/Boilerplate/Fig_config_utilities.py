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
gene_expression = pd.read_csv(('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/Geneexpression (full).tsv'), sep='\t', header=0)
tf_expression = pd.read_csv(('/home/christianl/Zhang-Lab/Zhang Lab Data/Full data files/TF(full).tsv'), sep='\t', header=0)

# Split into training, testing and validation sets and into numpy arrays + combining dataframes
x = tf_expression
y = gene_expression
# First split: 70% train and 30% temp (test + val)
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=42)

# Second split: split the temp set into 20% test and 10% val (which is 2/3 and 1/3 of temp)
x_test, x_val, y_test, y_val = train_test_split(
    x_temp, y_temp, test_size=1/3, random_state=42)

# For training set
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

# For validation set
x_val = x_val.to_numpy()
y_val = y_val.to_numpy()

# For testing set
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

###############################################################################################

#### centering script 
# column-wise centering for training set (each gene is a column, each row is an instance)
x_train_col_means = x_train.mean(axis=0)
x_train_centered = x_train - x_train_col_means

y_train_col_means = y_train.mean(axis=0)
y_train_centered = y_train - y_train_col_means

# for test set 
x_test_centered = x_test - x_train_col_means
y_test_centered = y_test - y_train_col_means

# for val set
x_val_centered = x_val - x_train_col_means
y_val_centered = y_val - y_train_col_means

###############################################################################################

##### loading MLR model (v2), extracting mlr_y_pred
mlr_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/MLR/MLR_v2/MLR_model_v2.joblib'
reg_loaded = joblib.load(mlr_model_path)
mlr_y_pred = reg_loaded.predict(x_test_centered)          
print(type(mlr_y_pred), mlr_y_pred.shape)

##### loading XGBRF models (v1)
xgbrf_model_path = '/home/christianl/Zhang-Lab/Zhang Lab Data/Saved models/Random Forest/Saved_Models_XGBRF_v1.pkl'
# find all saved models, compute xgbrf_y_pred (trained on uncentered data unlike MLR
# so need to keep it as x_test to avoid destroying performance) --> ultimately decided 
# to universally use centered data and just retrain XGBRFRegressor() on x_test_centered
with open(xgbrf_model_path, 'rb') as f:
    models = pickle.load(f)
xgbrf_y_pred = np.column_stack([model.predict(x_test_centered) for model in models])

####################################################################################################

##### assemble all y_pred into dictionary

predictions = {
    "MLR": mlr_y_pred,
    "XGBRFRegressor": xgbrf_y_pred,
}

# performance metrics, globally and at per-gene resolution, for further analysis  

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
###############################################################################################