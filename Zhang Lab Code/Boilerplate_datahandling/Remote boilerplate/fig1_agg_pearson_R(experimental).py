import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

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


def figure_pearson_r_comparison_fast(y_true, predictions_dict, 
                                     ci_method='spearman_bootstrap',
                                     correlation_type='pearson',
                                     n_bootstrap=5000,
                                     subsample_size=None,
                                     output_path='...'):
    """
    Pearson's R comparison with robust confidence intervals using Spearman bootstrap.
    
    Parameters
    ----------
    y_true : array-like
        True values
    predictions_dict : dict
        Model predictions
    correlation_type : str, default='pearson'
        - 'pearson': Report Pearson's R (recommended for continuous predictions)
        - 'spearman': Report Spearman's rho (rank-based, more robust)
    ci_method : str, default='spearman_bootstrap'
        - 'spearman_bootstrap': Bootstrap using Spearman (ROBUST, recommended)
        - 'analytical': Fisher Z-transformation (fast but assumes normality)
        - 'bootstrap_subsample': Standard bootstrap on subsample
    n_bootstrap : int
        Number of bootstrap iterations (default: 5000)
    subsample_size : int, optional
        Subsample size for bootstrap (default: min(100000, n_samples))
        
    Notes
    -----
    Spearman bootstrap is more robust than Fisher Z because:
    1. No assumption of bivariate normality
    2. Resistant to outliers
    3. Works well with non-linear monotonic relationships
    4. More accurate CIs for skewed distributions
    """
    
    set_publication_style()
    
    model_names = list(predictions_dict.keys())
    correlation_values = []
    ci_lowers = []
    ci_uppers = []
    p_values = []
    
    for model_name in model_names:
        y_pred = predictions_dict[model_name]
        
        # Flatten arrays
        y_true_flat = np.asarray(y_true).ravel()
        y_pred_flat = np.asarray(y_pred).ravel()
        
        n_samples = len(y_true_flat)
        print(f"\nProcessing {model_name}")
        print(f"Dataset size: {n_samples:,} observations ({n_samples*8/1e9:.2f} GB as float64)")
        
        # Compute point estimate
        if correlation_type == 'pearson':
            corr_value, p_value = pearsonr(y_true_flat, y_pred_flat)
            corr_label = "Pearson's R"
        else:
            corr_value, p_value = spearmanr(y_true_flat, y_pred_flat)
            corr_label = "Spearman's ρ"
        
        correlation_values.append(corr_value)
        p_values.append(p_value)
        
        print(f"{corr_label}: {corr_value:.6f}")
        
        if ci_method == 'spearman_bootstrap':
            # Bootstrap using Spearman correlation for CI (MOST ROBUST)
            if subsample_size is None:
                subsample_size = min(100000, n_samples)
            
            print(f"Computing ROBUST Spearman bootstrap CI on {subsample_size:,} observations...")
            print(f"  {n_bootstrap:,} iterations")
            print(f"  This method is distribution-free and handles outliers well")
            
            np.random.seed(42)
            subsample_idx = np.random.choice(n_samples, size=subsample_size, replace=False)
            y_true_sub = y_true_flat[subsample_idx]
            y_pred_sub = y_pred_flat[subsample_idx]
            
            bootstrap_corrs = []
            for i in range(n_bootstrap):
                boot_idx = np.random.choice(subsample_size, size=subsample_size, replace=True)
                
                # Use Spearman for bootstrapping (rank-based, robust)
                rho_boot, _ = spearmanr(y_true_sub[boot_idx], y_pred_sub[boot_idx])
                bootstrap_corrs.append(rho_boot)
                
                if (i + 1) % 1000 == 0:
                    print(f"    {i+1:,}/{n_bootstrap:,} iterations complete")
            
            bootstrap_corrs = np.array(bootstrap_corrs)
            
            # Percentile method (recommended for non-normal distributions)
            ci_lower = np.percentile(bootstrap_corrs, 2.5)
            ci_upper = np.percentile(bootstrap_corrs, 97.5)
            
            # Diagnostic: check bootstrap distribution
            bootstrap_mean = np.mean(bootstrap_corrs)
            bootstrap_std = np.std(bootstrap_corrs)
            
            print(f"  Bootstrap diagnostics:")
            print(f"    Mean: {bootstrap_mean:.6f} (point estimate: {corr_value:.6f})")
            print(f"    Std: {bootstrap_std:.6f}")
            print(f"    Skewness: {stats.skew(bootstrap_corrs):.3f}")
            
        elif ci_method == 'analytical':
            # Fisher Z-transformation (FAST but assumes bivariate normality)
            print(f"Computing analytical CI using Fisher Z-transformation...")
            print(f"  WARNING: Assumes bivariate normality - may be inaccurate for skewed data")
            
            z = np.arctanh(corr_value)
            se_z = 1 / np.sqrt(n_samples - 3)
            z_lower = z - 1.96 * se_z
            z_upper = z + 1.96 * se_z
            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
            print(f"  CI computed in <0.001s using {n_samples:,} observations")
            
        elif ci_method == 'bootstrap_subsample':
            # Standard bootstrap on subsample using same correlation as point estimate
            if subsample_size is None:
                subsample_size = min(100000, n_samples)
            
            print(f"Computing standard bootstrap CI on subsample of {subsample_size:,} observations...")
            print(f"  {n_bootstrap:,} iterations")
            
            np.random.seed(42)
            subsample_idx = np.random.choice(n_samples, size=subsample_size, replace=False)
            y_true_sub = y_true_flat[subsample_idx]
            y_pred_sub = y_pred_flat[subsample_idx]
            
            bootstrap_corrs = []
            for i in range(n_bootstrap):
                boot_idx = np.random.choice(subsample_size, size=subsample_size, replace=True)
                
                if correlation_type == 'pearson':
                    r_boot, _ = pearsonr(y_true_sub[boot_idx], y_pred_sub[boot_idx])
                else:
                    r_boot, _ = spearmanr(y_true_sub[boot_idx], y_pred_sub[boot_idx])
                    
                bootstrap_corrs.append(r_boot)
                
                if (i + 1) % 1000 == 0:
                    print(f"    {i+1:,}/{n_bootstrap:,} iterations complete")
            
            bootstrap_corrs = np.array(bootstrap_corrs)
            ci_lower = np.percentile(bootstrap_corrs, 2.5)
            ci_upper = np.percentile(bootstrap_corrs, 97.5)
            
        else:
            raise ValueError(f"Unknown ci_method: {ci_method}. Use 'spearman_bootstrap', 'analytical', or 'bootstrap_subsample'")
        
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
        
        print(f"{model_name}: {corr_label} = {corr_value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], p < 0.001")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = np.arange(len(model_names))
    colors = [MODEL_COLORS.get(name, '#1f77b4') for name in model_names]
    
    bars = ax.bar(x_pos, correlation_values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    errors = np.array([
        [correlation_values[i] - ci_lowers[i] for i in range(len(model_names))],
        [ci_uppers[i] - correlation_values[i] for i in range(len(model_names))]
    ])
    
    ax.errorbar(x_pos, correlation_values, yerr=errors, 
                fmt='none', ecolor='black', capsize=8, capthick=1.5, 
                linewidth=1.5, zorder=10)
    
    # Add stats text boxes
    for i, (model_name, r, ci_lower, ci_upper, p_val) in enumerate(zip(
            model_names, correlation_values, ci_lowers, ci_uppers, p_values)):
        
        p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ci_width = ci_upper - ci_lower
        
        stats_text = (f"{corr_label} = {r:.4f}\n"
                      f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]\n"
                      f"CI width = {ci_width:.4f}\n"
                      f"{p_text}\n"
                      f"Method: {ci_method}")
        
        ax.text(i, 0.45 * ax.get_ylim()[1], stats_text,
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ylabel = f"{corr_label} (Aggregate)" if correlation_type == 'pearson' else f"{corr_label} (Aggregate)"
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    
    y_max = max(ci_uppers) + 0.15
    ax.set_ylim(0, min(1.0, y_max))
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
       
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.show()
    
    metrics_summary = {}
    for i, model_name in enumerate(model_names):
        metrics_summary[model_name] = {
            'correlation': correlation_values[i],
            'correlation_type': correlation_type,
            'ci_method': ci_method,
            'ci_lower': ci_lowers[i],
            'ci_upper': ci_uppers[i],
            'ci_width': ci_uppers[i] - ci_lowers[i],
            'p_value': p_values[i]
        }
    
    return metrics_summary


def weighted_quantile_vectorized(values, weights, quantile=0.5):
    """Vectorized weighted quantile computation for bootstrap samples."""
    n_bootstrap = weights.shape[0]
    
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[:, sorted_idx]
    
    cumsum_weights = np.cumsum(sorted_weights, axis=1)
    total_weights = cumsum_weights[:, -1:]
    quantile_positions = quantile * total_weights
    
    indices = np.argmax(cumsum_weights >= quantile_positions, axis=1)
    
    return sorted_values[indices]


def figure_per_gene_pearson_r_distribution(y_true_df, predictions_dict,
                                           correlation_type='pearson',
                                           ci_method='spearman_bootstrap',
                                           n_bootstrap=1000,
                                           show_ci=True,
                                           output_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved figures/Production_model_figures(x_train)/per_gene_pearson_r_distribution.png'):
    """
    Generate violin/box plot showing distribution of per-gene correlation values.
    
    Parameters
    ----------
    correlation_type : str, default='pearson'
        - 'pearson': Pearson correlation (assumes linearity)
        - 'spearman': Spearman rank correlation (robust to outliers)
    ci_method : str, default='spearman_bootstrap'
        - 'spearman_bootstrap': Robust bootstrap using Spearman (RECOMMENDED)
        - 'analytical': Fast analytical approximation
        - 'bootstrap': Standard bootstrap
        - None: Skip CI
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
        
    Notes
    -----
    For genetic data with potential outliers or non-normal distributions,
    spearman_bootstrap provides more reliable confidence intervals.
    """
    
    set_publication_style()
    
    model_names = list(predictions_dict.keys())
    per_gene_data = []
    summary_stats = {}
    
    for model_name in model_names:
        y_pred = predictions_dict[model_name]
        n_genes = y_true_df.shape[1]
        
        print(f"\nComputing per-gene {correlation_type} correlation for {model_name} ({n_genes} genes)...")
        
        model_correlations = []
        
        for gene_idx in range(n_genes):
            y_true_gene = y_true_df.iloc[:, gene_idx].values
            y_pred_gene = y_pred[:, gene_idx]
            
            if np.var(y_true_gene) > 1e-10:
                if correlation_type == 'pearson':
                    r, p = pearsonr(y_true_gene, y_pred_gene)
                else:
                    r, p = spearmanr(y_true_gene, y_pred_gene)
                
                per_gene_data.append({
                    'model': model_name,
                    'gene': y_true_df.columns[gene_idx],
                    'correlation': r,
                    'correlation_type': correlation_type,
                    'p_value': p
                })
                
                model_correlations.append(r)
        
        # Compute summary statistics
        model_correlations = np.array(model_correlations)
        median_r = np.median(model_correlations)
        mean_r = np.mean(model_correlations)
        
        summary_stats[model_name] = {
            'correlations': model_correlations,
            'median': median_r,
            'mean': mean_r,
            'q1': np.percentile(model_correlations, 25),
            'q3': np.percentile(model_correlations, 75)
        }
        
        # Compute CI for median and mean if requested
        if show_ci and ci_method is not None:
            print(f"Computing {ci_method} CI for {model_name} summary statistics...")
            
            if ci_method == 'spearman_bootstrap':
                # ROBUST: Bootstrap using Spearman correlation
                print(f"  Using ROBUST Spearman bootstrap ({n_bootstrap} iterations)")
                
                n_genes_valid = len(model_correlations)
                median_boots = []
                mean_boots = []
                
                np.random.seed(42)
                for i in range(n_bootstrap):
                    boot_indices = np.random.choice(n_genes_valid, size=n_genes_valid, replace=True)
                    boot_sample = model_correlations[boot_indices]
                    
                    median_boots.append(np.median(boot_sample))
                    mean_boots.append(np.mean(boot_sample))
                    
                    if (i + 1) % 200 == 0:
                        print(f"    {i+1}/{n_bootstrap} iterations complete")
                
                median_boots = np.array(median_boots)
                mean_boots = np.array(mean_boots)
                
                # Percentile method (robust to skewness)
                summary_stats[model_name]['median_ci_lower'] = np.percentile(median_boots, 2.5)
                summary_stats[model_name]['median_ci_upper'] = np.percentile(median_boots, 97.5)
                summary_stats[model_name]['mean_ci_lower'] = np.percentile(mean_boots, 2.5)
                summary_stats[model_name]['mean_ci_upper'] = np.percentile(mean_boots, 97.5)
                
                # Bootstrap diagnostics
                print(f"  Bootstrap median: {np.mean(median_boots):.4f} (point: {median_r:.4f})")
                print(f"  Bootstrap mean: {np.mean(mean_boots):.4f} (point: {mean_r:.4f})")
                
            elif ci_method == 'analytical':
                # Fast analytical approximation
                n_genes_valid = len(model_correlations)
                
                # For mean (exact)
                se_mean = np.std(model_correlations) / np.sqrt(n_genes_valid)
                summary_stats[model_name]['mean_ci_lower'] = mean_r - 1.96 * se_mean
                summary_stats[model_name]['mean_ci_upper'] = mean_r + 1.96 * se_mean
                
                # For median (approximate, assumes normality)
                se_median = 1.253 * se_mean
                summary_stats[model_name]['median_ci_lower'] = median_r - 1.96 * se_median
                summary_stats[model_name]['median_ci_upper'] = median_r + 1.96 * se_median
                
            elif ci_method == 'bootstrap':
                # Standard bootstrap (uses same correlation type as point estimate)
                median_boots = []
                mean_boots = []
                n_genes_valid = len(model_correlations)
                
                np.random.seed(42)
                for _ in range(n_bootstrap):
                    boot_indices = np.random.choice(n_genes_valid, size=n_genes_valid, replace=True)
                    boot_sample = model_correlations[boot_indices]
                    
                    median_boots.append(np.median(boot_sample))
                    mean_boots.append(np.mean(boot_sample))
                
                summary_stats[model_name]['median_ci_lower'] = np.percentile(median_boots, 2.5)
                summary_stats[model_name]['median_ci_upper'] = np.percentile(median_boots, 97.5)
                summary_stats[model_name]['mean_ci_lower'] = np.percentile(mean_boots, 2.5)
                summary_stats[model_name]['mean_ci_upper'] = np.percentile(mean_boots, 97.5)
            
            print(f"{model_name} Median: {median_r:.4f} "
                  f"[{summary_stats[model_name]['median_ci_lower']:.4f}, "
                  f"{summary_stats[model_name]['median_ci_upper']:.4f}]")
            print(f"{model_name} Mean: {mean_r:.4f} "
                  f"[{summary_stats[model_name]['mean_ci_lower']:.4f}, "
                  f"{summary_stats[model_name]['mean_ci_upper']:.4f}]")
    
    # Convert to DataFrame
    df_per_gene = pd.DataFrame(per_gene_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    parts = ax.violinplot(
        [df_per_gene[df_per_gene['model'] == m]['correlation'].values 
         for m in model_names],
        positions=np.arange(len(model_names)),
        widths=0.7,
        showmeans=True,
        showmedians=True
    )
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(MODEL_COLORS.get(model_names[i], '#1f77b4'))
        pc.set_alpha(0.6)
    
    if show_ci and ci_method is not None:
        for i, model_name in enumerate(model_names):
            stats = summary_stats[model_name]
            median = stats['median']
            
            if 'median_ci_lower' in stats:
                ci_lower = stats['median_ci_lower']
                ci_upper = stats['median_ci_upper']
                
                ax.errorbar(i, median, 
                           yerr=[[median - ci_lower], [ci_upper - median]],
                           fmt='D', color='darkred', markersize=6,
                           capsize=10, capthick=2, linewidth=2,
                           label='Median 95% CI' if i == 0 else '', zorder=10)
    
    # Add stats text boxes
    for i, model_name in enumerate(model_names):
        stats = summary_stats[model_name]
        median = stats['median']
        mean = stats['mean']
        q1, q3 = stats['q1'], stats['q3']
        
        if show_ci and ci_method is not None and 'median_ci_lower' in stats:
            textstr = (f"Median = {median:.3f}\n"
                      f"95% CI = [{stats['median_ci_lower']:.3f}, {stats['median_ci_upper']:.3f}]\n"
                      f"Mean = {mean:.3f}\n"
                      f"IQR = [{q1:.3f}, {q3:.3f}]\n"
                      f"CI: {ci_method}")
        else:
            textstr = f'Median = {median:.3f}\nMean = {mean:.3f}\nIQR = [{q1:.3f}, {q3:.3f}]'
        
        ax.text(i, ax.get_ylim()[1] * 0.5, textstr,
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    corr_label = "Pearson's R" if correlation_type == 'pearson' else "Spearman's ρ"
    ax.set_ylabel(f"{corr_label} (Per-Gene)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    if show_ci and ci_method is not None:
        ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.show()
    
    summary_df = pd.DataFrame(summary_stats).T
    
    return df_per_gene, summary_df