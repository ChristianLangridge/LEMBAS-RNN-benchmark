import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

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

# getting y_test and each model's y_pred  

##### loading MLR model (v2), extracting mlr_y_pred
mlr_model_path = "~/Zhang-Lab/Zhang Lab Data/Saved models/MLR/MLR_model_v2.joblib"
reg_loaded = joblib.load(model_path)
mlr_y_pred = reg_loaded.predict(x_test)          
print(type(mlr_y_pred), mlr_y_pred.shape)

##### loading XGBRF models (v1), extracting xgbrf_y_pred

xgbrf_model_path = '/Users/christianlangridge/Desktop/Zhang-Lab/Zhang Lab Data/Saved models/Random Forest/Saved_Models_XGBRF_v2'

# Find all saved models; ensure consistent order

model_paths = sorted(glob.glob(os.path.join(model_dir, "target_*.json")))
models = []
for path in model_paths:
    est = xgb.XGBRFRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100,
        max_depth=5,
        device='cuda',
        tree_method='hist'
    )
    est.load_model(path)
    models.append(est)
print(f"Loaded {len(models)} models from {model_dir}")

xgbrf_y_pred = models.predict(x_test)          
print(type(xgbrf_y_pred), xgbrf_y_pred.shape)


predictions = {
    "MLR": mlr_y_pred,
    "XGBRFRegressor": xgbrf_y_pred,
}












# performance metrics for further analysis   
def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive performance metrics.
    
    Parameters
    ----------
    y_true : array-like
        Observed expression values
    y_pred : array-like
        Predicted expression values
    
    Returns
    -------
    dict : Dictionary containing RÂ², RMSE, MAE, Pearson r, and p-value
    """
    pearson_r, p_value = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'r2': r2,
        'pearson_r': pearson_r,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae
    }
