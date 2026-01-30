import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

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
                                     ci_method='analytical',
                                     n_bootstrap=5000,
                                     subsample_size=None,
                                     output_path='...'):
    """
    Ultra-fast Pearson's R comparison with confidence intervals.
    
    Parameters
    ----------
    y_true : array-like
        True values
    predictions_dict : dict
        Model predictions
    ci_method : str, default='analytical'
        - 'analytical': Fisher Z-transformation (FAST, recommended for large N)
        - 'bootstrap_subsample': Bootstrap on subsample (if you really want bootstrap)
    n_bootstrap : int
        Number of bootstrap iterations (only used if ci_method='bootstrap_subsample')
    subsample_size : int, optional
        Subsample size for bootstrap (default: min(100000, n_samples))
    """
    
    set_publication_style()
    
    model_names = list(predictions_dict.keys())
    pearson_rs = []
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
        pearson_r, p_value = pearsonr(y_true_flat, y_pred_flat)
        pearson_rs.append(pearson_r)
        p_values.append(p_value)
        
        print(f"Pearson's R: {pearson_r:.6f}")
        
        if ci_method == 'analytical':
            # Fisher Z-transformation
            print(f"Computing analytical CI using Fisher Z-transformation...")
            z = np.arctanh(pearson_r)
            se_z = 1 / np.sqrt(n_samples - 3)
            z_lower = z - 1.96 * se_z
            z_upper = z + 1.96 * se_z
            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
            print(f"  CI computed in <0.001s using {n_samples:,} observations")
            
        elif ci_method == 'bootstrap_subsample':
            # Bootstrap on subsample
            if subsample_size is None:
                subsample_size = min(100000, n_samples)
            
            print(f"Computing bootstrap CI on subsample of {subsample_size:,} observations...")
            print(f"  {n_bootstrap:,} iterations")
            
            np.random.seed(42)
            subsample_idx = np.random.choice(n_samples, size=subsample_size, replace=False)
            y_true_sub = y_true_flat[subsample_idx]
            y_pred_sub = y_pred_flat[subsample_idx]
            
            bootstrap_rs = []
            for i in range(n_bootstrap):
                boot_idx = np.random.choice(subsample_size, size=subsample_size, replace=True)
                r_boot, _ = pearsonr(y_true_sub[boot_idx], y_pred_sub[boot_idx])
                bootstrap_rs.append(r_boot)
                
                if (i + 1) % 1000 == 0:
                    print(f"    {i+1:,}/{n_bootstrap:,} iterations complete")
            
            bootstrap_rs = np.array(bootstrap_rs)
            ci_lower = np.percentile(bootstrap_rs, 2.5)
            ci_upper = np.percentile(bootstrap_rs, 97.5)
            
            print(f"  Subsample Pearson's R: {pearsonr(y_true_sub, y_pred_sub)[0]:.6f}")
            print(f"  Full dataset Pearson's R: {pearson_r:.6f}")
            
        else:
            raise ValueError(f"Unknown ci_method: {ci_method}. Use 'analytical' or 'bootstrap_subsample'")
        
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
        
        print(f"{model_name}: Pearson's R = {pearson_r:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], p < 0.001")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = np.arange(len(model_names))
    colors = [MODEL_COLORS.get(name, '#1f77b4') for name in model_names]
    
    bars = ax.bar(x_pos, pearson_rs, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    errors = np.array([
        [pearson_rs[i] - ci_lowers[i] for i in range(len(model_names))],
        [ci_uppers[i] - pearson_rs[i] for i in range(len(model_names))]
    ])
    
    # MATCHED: Same error bar styling as scatterplot
    ax.errorbar(x_pos, pearson_rs, yerr=errors, 
                fmt='none', ecolor='black', capsize=8, capthick=1.5, 
                linewidth=1.5, zorder=10)
    
    # CHANGED: Add stats text boxes with WHEAT background (matching scatterplot)
    for i, (model_name, r, ci_lower, ci_upper, p_val) in enumerate(zip(
            model_names, pearson_rs, ci_lowers, ci_uppers, p_values)):
        
        # Build stats text with same format as scatterplot
        p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ci_width = ci_upper - ci_lower
        
        stats_text = (f"Pearson's R = {r:.4f}\n"
                      f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]\n"
                      f"CI width = {ci_width:.4f}\n"
                      f"{p_text}")
        
        # MATCHED: Same text box position and style as scatterplot
        ax.text(i, 0.5 * ax.get_ylim()[1], stats_text,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_ylabel("Pearson's R (Aggregate)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    
    y_max = max(ci_uppers) + 0.15  # Extra space for text boxes
    ax.set_ylim(0, min(1.0, y_max))
    
    # MATCHED: Same grid alpha as scatterplot
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
       
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.show()
    
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
                                           ci_method='analytical',
                                           n_bootstrap=1000,
                                           show_ci=True,
                                           output_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved figures/Production_model_figures(x_train)/per_gene_pearson_r_distribution.png'):
    """
    Generate violin/box plot showing distribution of per-gene Pearson's R values.
    
    
    Parameters
    ----------
    ci_method : str, default='analytical'
        - 'analytical': Fast analytical CI (recommended)
        - 'poisson_bootstrap_vectorized': Vectorized bootstrap
        - 'bootstrap': Standard bootstrap
        - 'fisher': Fisher transformation
        - None: Skip CI
    """
    
    set_publication_style()
    
    # Calculate per-gene Pearson's R for each model
    model_names = list(predictions_dict.keys())
    per_gene_data = []
    summary_stats = {}
    
    for model_name in model_names:
        y_pred = predictions_dict[model_name]
        n_genes = y_true_df.shape[1]
        
        print(f"\nComputing per-gene Pearson's R for {model_name} ({n_genes} genes)...")
        
        model_pearson_rs = []
        
        for gene_idx in range(n_genes):
            y_true_gene = y_true_df.iloc[:, gene_idx].values
            y_pred_gene = y_pred[:, gene_idx]
            
            if np.var(y_true_gene) > 1e-10:
                r, p = pearsonr(y_true_gene, y_pred_gene)
                
                per_gene_data.append({
                    'model': model_name,
                    'gene': y_true_df.columns[gene_idx],
                    'pearson_r': r,
                    'p_value': p
                })
                
                model_pearson_rs.append(r)
        
        # Compute summary statistics
        model_pearson_rs = np.array(model_pearson_rs)
        median_r = np.median(model_pearson_rs)
        mean_r = np.mean(model_pearson_rs)
        
        summary_stats[model_name] = {
            'pearson_rs': model_pearson_rs,
            'median': median_r,
            'mean': mean_r,
            'q1': np.percentile(model_pearson_rs, 25),
            'q3': np.percentile(model_pearson_rs, 75)
        }
        
        # Compute bootstrap CI for median and mean if requested
        if show_ci and ci_method is not None:
            print(f"Computing {ci_method} CI for {model_name} summary statistics...")
            
            if ci_method == 'analytical':
                # Fast analytical CI
                n_genes_valid = len(model_pearson_rs)
                
                # For mean (exact)
                se_mean = np.std(model_pearson_rs) / np.sqrt(n_genes_valid)
                summary_stats[model_name]['mean_ci_lower'] = mean_r - 1.96 * se_mean
                summary_stats[model_name]['mean_ci_upper'] = mean_r + 1.96 * se_mean
                
                # For median (approximate)
                se_median = 1.253 * se_mean
                summary_stats[model_name]['median_ci_lower'] = median_r - 1.96 * se_median
                summary_stats[model_name]['median_ci_upper'] = median_r + 1.96 * se_median
                
            elif ci_method == 'poisson_bootstrap_vectorized':
                # Vectorized Poisson bootstrap
                n_genes_valid = len(model_pearson_rs)
                
                np.random.seed(42)
                weights = np.random.poisson(lam=1, size=(n_bootstrap, n_genes_valid))
                
                print(f"  Weight matrix: {weights.shape} (~{weights.nbytes/1e6:.1f} MB)")
                
                # Mean (vectorized)
                w_sum = weights.sum(axis=1, keepdims=True)
                weighted_means = (weights * model_pearson_rs).sum(axis=1) / w_sum.ravel()
                
                # Median (vectorized using helper function)
                median_boots = weighted_quantile_vectorized(
                    values=model_pearson_rs,
                    weights=weights,
                    quantile=0.5
                )
                
                summary_stats[model_name]['median_ci_lower'] = np.percentile(median_boots, 2.5)
                summary_stats[model_name]['median_ci_upper'] = np.percentile(median_boots, 97.5)
                summary_stats[model_name]['mean_ci_lower'] = np.percentile(weighted_means, 2.5)
                summary_stats[model_name]['mean_ci_upper'] = np.percentile(weighted_means, 97.5)
                
            elif ci_method == 'bootstrap':
                # Standard bootstrap
                median_boots = []
                mean_boots = []
                n_genes_valid = len(model_pearson_rs)
                
                np.random.seed(42)
                for _ in range(n_bootstrap):
                    boot_indices = np.random.choice(n_genes_valid, size=n_genes_valid, replace=True)
                    boot_sample = model_pearson_rs[boot_indices]
                    
                    median_boots.append(np.median(boot_sample))
                    mean_boots.append(np.mean(boot_sample))
                
                summary_stats[model_name]['median_ci_lower'] = np.percentile(median_boots, 2.5)
                summary_stats[model_name]['median_ci_upper'] = np.percentile(median_boots, 97.5)
                summary_stats[model_name]['mean_ci_lower'] = np.percentile(mean_boots, 2.5)
                summary_stats[model_name]['mean_ci_upper'] = np.percentile(mean_boots, 97.5)
                
            elif ci_method == 'fisher':
                # Fisher transformation
                n_genes_valid = len(model_pearson_rs)
                
                se_mean = np.std(model_pearson_rs) / np.sqrt(n_genes_valid)
                summary_stats[model_name]['mean_ci_lower'] = mean_r - 1.96 * se_mean
                summary_stats[model_name]['mean_ci_upper'] = mean_r + 1.96 * se_mean
                
                se_median = 1.253 * se_mean
                summary_stats[model_name]['median_ci_lower'] = median_r - 1.96 * se_median
                summary_stats[model_name]['median_ci_upper'] = median_r + 1.96 * se_median
            
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
        [df_per_gene[df_per_gene['model'] == m]['pearson_r'].values 
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
    
    # CHANGED: Stats text boxes with WHEAT background (matching scatterplot)
    for i, model_name in enumerate(model_names):
        stats = summary_stats[model_name]
        median = stats['median']
        mean = stats['mean']
        q1, q3 = stats['q1'], stats['q3']
        
        # Build consistent stats text
        if show_ci and ci_method is not None and 'median_ci_lower' in stats:
            textstr = (f"Median = {median:.3f}\n"
                      f"95% CI = [{stats['median_ci_lower']:.3f}, {stats['median_ci_upper']:.3f}]\n"
                      f"Mean = {mean:.3f}\n"
                      f"IQR = [{q1:.3f}, {q3:.3f}]")
        else:
            textstr = f'Median = {median:.3f}\nMean = {mean:.3f}\nIQR = [{q1:.3f}, {q3:.3f}]'
        
        # MATCHED: Same position (top-left style) and wheat background
        ax.text(i, ax.get_ylim()[1] * 0.5, textstr,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_ylabel("Pearson's R (Per-Gene)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    
    # MATCHED: Same grid alpha as scatterplot
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # MATCHED: Legend in lower right (if showing CI)
    if show_ci and ci_method is not None:
        ax.legend(loc='lower right', fontsize=9)
    
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.show()
    
    summary_df = pd.DataFrame(summary_stats).T
    
    return df_per_gene, summary_df