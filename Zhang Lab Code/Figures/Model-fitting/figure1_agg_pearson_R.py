import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

# Assuming MODEL_COLORS, DPI, and set_publication_style are imported from your config file
# If running standalone, uncomment these:

MODEL_COLORS = {
    'RNN': '#1f77b4',           # Deep blue
    'XGBRFRegressor': '#ff7f0e', # Orange
    'MLR': '#2ca02c'          # Green
}
DPI = 300

def set_publication_style():
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


def figure_pearson_r_comparison(y_true, predictions_dict, 
                                 ci_method='bootstrap',
                                 n_bootstrap=10000,
                                 output_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved figures/Production_model_figures(x_train)/pearson_r_comparison.png'):
    """
    Generate bar plot comparing aggregate Pearson's R across models with 95% CI.
    
    This function computes Pearson's R on flattened predictions (treating all 16k+ 
    prediction tasks equally) and estimates 95% confidence intervals using either 
    bootstrap resampling or Fisher's z-transformation.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_genes)
        True target gene expression values (ground truth)
    predictions_dict : dict
        Dictionary with keys as model names and values as predictions
        Example: {'RNN': pred_rnn, 'XGBRFRegressor': pred_xgb, 'MLR': pred_mlr}
    ci_method : str, default='bootstrap'
        Method for computing 95% confidence intervals:
        - 'bootstrap': Non-parametric bootstrap resampling (recommended for large datasets)
        - 'fisher': Fisher's z-transformation (parametric, assumes bivariate normality)
    n_bootstrap : int, default=10000
        Number of bootstrap iterations (only used if ci_method='bootstrap')
    output_path : str
        Path to save the figure
        
    Returns
    -------
    metrics_summary : dict
        Dictionary containing Pearson's R, 95% CI bounds, and p-values for each model
        
    Notes
    -----
    - Pearson's R is computed on flattened data (all genes treated equally)
    - Bootstrap CI provides non-parametric estimates robust to non-normality
    - Fisher's z-transformation assumes bivariate normality but is faster
    - For 16k+ genes with ~1000s samples, bootstrap is computationally feasible
    
    Examples
    --------
    >>> metrics = figure_pearson_r_comparison(y_test, predictions, 
    ...                                       ci_method='bootstrap',
    ...                                       n_bootstrap=10000)
    >>> print(f"MLR Pearson's R: {metrics['MLR']['pearson_r']:.4f} "
    ...       f"[{metrics['MLR']['ci_lower']:.4f}, {metrics['MLR']['ci_upper']:.4f}]")
    """
    
    set_publication_style()
    
    # Calculate Pearson's R and confidence intervals for each model
    model_names = list(predictions_dict.keys())
    pearson_rs = []
    ci_lowers = []
    ci_uppers = []
    p_values = []
    
    for model_name in model_names:
        y_pred = predictions_dict[model_name]
        
        # Flatten arrays (treating all genes equally)
        y_true_flat = np.asarray(y_true).ravel()
        y_pred_flat = np.asarray(y_pred).ravel()
        
        # Compute Pearson's R
        pearson_r, p_value = pearsonr(y_true_flat, y_pred_flat)
        pearson_rs.append(pearson_r)
        p_values.append(p_value)
        
        # Compute 95% confidence interval
        if ci_method == 'bootstrap':
            # Bootstrap resampling for CI estimation
            print(f"Computing bootstrap CI for {model_name} ({n_bootstrap} iterations)...")
            bootstrap_rs = []
            n_samples = len(y_true_flat)
            
            np.random.seed(42)  # For reproducibility
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                y_true_boot = y_true_flat[indices]
                y_pred_boot = y_pred_flat[indices]
                
                # Compute Pearson's R for this bootstrap sample
                r_boot, _ = pearsonr(y_true_boot, y_pred_boot)
                bootstrap_rs.append(r_boot)
            
            # 95% CI from bootstrap distribution (percentile method)
            ci_lower = np.percentile(bootstrap_rs, 2.5)
            ci_upper = np.percentile(bootstrap_rs, 97.5)
            
        elif ci_method == 'fisher':
            # Fisher's z-transformation for CI estimation
            # This assumes bivariate normality
            n = len(y_true_flat)
            
            # Fisher's z-transformation
            z = np.arctanh(pearson_r)
            
            # Standard error of z
            se_z = 1 / np.sqrt(n - 3)
            
            # 95% CI in z-space (z-score = 1.96 for 95% CI)
            z_lower = z - 1.96 * se_z
            z_upper = z + 1.96 * se_z
            
            # Transform back to correlation space
            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
            
        else:
            raise ValueError(f"Unknown ci_method: {ci_method}. Use 'bootstrap' or 'fisher'.")
        
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
        
        print(f"{model_name}: Pearson's R = {pearson_r:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], p < 0.001")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # X-axis positions for bars
    x_pos = np.arange(len(model_names))
    
    # Get colors for each model
    colors = [MODEL_COLORS.get(name, '#1f77b4') for name in model_names]
    
    # Create bars
    bars = ax.bar(x_pos, pearson_rs, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    # Add error bars (95% CI)
    errors = np.array([
        [pearson_rs[i] - ci_lowers[i] for i in range(len(model_names))],
        [ci_uppers[i] - pearson_rs[i] for i in range(len(model_names))]
    ])
    
    ax.errorbar(x_pos, pearson_rs, yerr=errors, 
                fmt='none', ecolor='black', capsize=8, capthick=1.5, 
                linewidth=1.5, zorder=10)
    
    # Add value labels on top of bars
    for i, (r, ci_lower, ci_upper) in enumerate(zip(pearson_rs, ci_lowers, ci_uppers)):
        # Main value
        ax.text(i, r + (ci_upper - r) + 0.01, f'{r:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # CI range (optional, can be commented out if too cluttered)
        # ax.text(i, r - 0.02, f'[{ci_lower:.3f}, {ci_upper:.3f}]', 
        #         ha='center', va='top', fontsize=8, style='italic')
    
    # Formatting
    ax.set_ylabel("Pearson's R (Aggregate)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_title("Model Performance Comparison: Aggregate Pearson's R with 95% CI", 
                 fontsize=13, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    
    # Y-axis limits and grid
    y_max = max(ci_uppers) + 0.05
    ax.set_ylim(0, min(1.0, y_max))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add reference line at R=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add subtitle with method info
    method_text = f"95% CI estimated via {ci_method}" + \
                  (f" ({n_bootstrap:,} iterations)" if ci_method == 'bootstrap' else "")
    fig.text(0.5, 0.02, method_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.show()
    
    # Return metrics summary
    metrics_summary = {}
    for i, model_name in enumerate(model_names):
        metrics_summary[model_name] = {
            'pearson_r': pearson_rs[i],
            'ci_lower': ci_lowers[i],
            'ci_upper': ci_uppers[i],
            'ci_width': ci_uppers[i] - ci_lowers[i],
            'p_value': p_values[i]
        }
    
    return metrics_summary


# Alternative version: Per-gene Pearson's R distribution
def figure_per_gene_pearson_r_distribution(y_true_df, predictions_dict,
                                           output_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved figures/Production_model_figures(x_train)/per_gene_pearson_r_distribution.png'):
    """
    Generate violin/box plot showing distribution of per-gene Pearson's R values.
    
    This provides a complementary view showing how consistent model performance
    is across the 16k+ individual prediction tasks.
    
    Parameters
    ----------
    y_true_df : pd.DataFrame, shape (n_samples, n_genes)
        True target gene expression values with gene names as columns
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values
    output_path : str
        Path to save the figure
        
    Returns
    -------
    per_gene_metrics : dict
        Dictionary containing per-gene Pearson's R values for each model
    """
    
    set_publication_style()
    
    # Calculate per-gene Pearson's R for each model
    model_names = list(predictions_dict.keys())
    per_gene_data = []
    
    for model_name in model_names:
        y_pred = predictions_dict[model_name]
        n_genes = y_true_df.shape[1]
        
        print(f"Computing per-gene Pearson's R for {model_name} ({n_genes} genes)...")
        
        for gene_idx in range(n_genes):
            y_true_gene = y_true_df.iloc[:, gene_idx].values
            y_pred_gene = y_pred[:, gene_idx]
            
            # Only compute if gene has variance
            if np.var(y_true_gene) > 1e-10:
                r, p = pearsonr(y_true_gene, y_pred_gene)
                
                per_gene_data.append({
                    'model': model_name,
                    'gene': y_true_df.columns[gene_idx],
                    'pearson_r': r,
                    'p_value': p
                })
    
    # Convert to DataFrame
    df_per_gene = pd.DataFrame(per_gene_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Violin plot with box plot overlay
    parts = ax.violinplot(
        [df_per_gene[df_per_gene['model'] == m]['pearson_r'].values 
         for m in model_names],
        positions=np.arange(len(model_names)),
        widths=0.7,
        showmeans=True,
        showmedians=True
    )
    
    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(MODEL_COLORS.get(model_names[i], '#1f77b4'))
        pc.set_alpha(0.6)
    
    # Add summary statistics
    for i, model_name in enumerate(model_names):
        data = df_per_gene[df_per_gene['model'] == model_name]['pearson_r'].values
        median = np.median(data)
        mean = np.mean(data)
        q1, q3 = np.percentile(data, [25, 75])
        
        # Add text annotation
        textstr = f'Median: {median:.3f}\nMean: {mean:.3f}\nIQR: [{q1:.3f}, {q3:.3f}]'
        ax.text(i, ax.get_ylim()[1] * 0.95, textstr,
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_ylabel("Pearson's R (Per-Gene)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_title(f"Distribution of Per-Gene Pearson's R ({len(df_per_gene[df_per_gene['model'] == model_names[0]]):,} genes)", 
                 fontsize=13, fontweight='bold', pad=20)
    
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.show()
    
    return df_per_gene


if __name__ == "__main__":
    """
    Example usage (requires data from your Fig-config_utilities script)
    
    # After loading predictions_dict and y_test from your config file:
    
    metrics = figure_pearson_r_comparison(
        y_true=y_test,
        predictions_dict=predictions,
        ci_method='bootstrap',
        n_bootstrap=10000,
        output_path='~/Zhang-Lab/Zhang Lab Data/Saved figures/pearson_r_comparison.png'
    )
    
    # Optional: Show per-gene distribution
    per_gene_df = figure_per_gene_pearson_r_distribution(
        y_true_df=y_test,
        predictions_dict=predictions,
        output_path='~/Zhang-Lab/Zhang Lab Data/Saved figures/per_gene_pearson_r_distribution.png'
    )
    """
    pass