import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import os
import sys
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


def figure_dual_correlation_comparison(y_true, predictions_dict,
                                      compute_spearman=True,
                                      spearman_subsample=500000,
                                      output_path='correlation_comparison.png'):
    """
    Compare models using BOTH Pearson and Spearman with Fisher Z CIs.
    
    This extends your existing figure_pearson_r_comparison_fast() to include
    Spearman as a robustness check, using Fisher Z for both metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    predictions_dict : dict
        Model predictions
    compute_spearman : bool, default=True
        Whether to compute Spearman alongside Pearson
    spearman_subsample : int, default=500000
        Subsample size for Spearman (ranking is O(N log N))
        Set to None to use full dataset
    output_path : str
        Where to save the figure
        
    Returns
    -------
    metrics_summary : dict
        Dictionary with Pearson (and optionally Spearman) metrics for each model
    """
    
    set_publication_style()
    
    model_names = list(predictions_dict.keys())
    
    # Storage for metrics
    pearson_rs = []
    pearson_ci_lowers = []
    pearson_ci_uppers = []
    p_values_pearson = []
    
    spearman_rhos = []
    spearman_ci_lowers = []
    spearman_ci_uppers = []
    p_values_spearman = []
    
    for model_name in model_names:
        y_pred = predictions_dict[model_name]
        
        # Flatten arrays
        y_true_flat = np.asarray(y_true).ravel()
        y_pred_flat = np.asarray(y_pred).ravel()
        
        n_samples = len(y_true_flat)
        print(f"\n{'='*70}")
        print(f"Processing {model_name}")
        print(f"Dataset size: {n_samples:,} observations ({n_samples*8/1e9:.2f} GB)")
        print(f"{'='*70}")
        
        # ================================================================
        # PEARSON (your original code)
        # ================================================================
        print("\n[1/2] Pearson correlation (full dataset)...")
        pearson_r, p_value = pearsonr(y_true_flat, y_pred_flat)
        pearson_rs.append(pearson_r)
        p_values_pearson.append(p_value)
        
        # Fisher Z-transformation
        z = np.arctanh(pearson_r)
        se_z = 1 / np.sqrt(n_samples - 3)
        z_lower = z - 1.96 * se_z
        z_upper = z + 1.96 * se_z
        ci_lower = np.tanh(z_lower)
        ci_upper = np.tanh(z_upper)
        
        pearson_ci_lowers.append(ci_lower)
        pearson_ci_uppers.append(ci_upper)
        
        print(f"  Pearson's R = {pearson_r:.6f}")
        print(f"  95% CI = [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"  CI width = {ci_upper - ci_lower:.6f}")
        
        # ================================================================
        # SPEARMAN (new, with same Fisher Z approach)
        # ================================================================
        if compute_spearman:
            print(f"\n[2/2] Spearman correlation...")
            
            # Decide whether to subsample
            if spearman_subsample and n_samples > spearman_subsample:
                print(f"  Using subsample of {spearman_subsample:,} observations")
                print(f"  (Ranking {n_samples:,} values is O(N log N), slow)")
                
                np.random.seed(42)
                subsample_idx = np.random.choice(n_samples, 
                                                size=spearman_subsample, 
                                                replace=False)
                y_true_sub = y_true_flat[subsample_idx]
                y_pred_sub = y_pred_flat[subsample_idx]
                
                spearman_rho, p_value_spear = spearmanr(y_true_sub, y_pred_sub)
                n_spear = spearman_subsample
            else:
                print(f"  Using FULL dataset ({n_samples:,} observations)")
                spearman_rho, p_value_spear = spearmanr(y_true_flat, y_pred_flat)
                n_spear = n_samples
            
            spearman_rhos.append(spearman_rho)
            p_values_spearman.append(p_value_spear)
            
            # Fisher Z for Spearman (works at large N!)
            z_spear = np.arctanh(spearman_rho)
            se_z_spear = 1 / np.sqrt(n_spear - 3)
            z_lower_spear = z_spear - 1.96 * se_z_spear
            z_upper_spear = z_spear + 1.96 * se_z_spear
            ci_lower_spear = np.tanh(z_lower_spear)
            ci_upper_spear = np.tanh(z_upper_spear)
            
            spearman_ci_lowers.append(ci_lower_spear)
            spearman_ci_uppers.append(ci_upper_spear)
            
            print(f"  Spearman's ρ = {spearman_rho:.6f}")
            print(f"  95% CI = [{ci_lower_spear:.6f}, {ci_upper_spear:.6f}]")
            print(f"  CI width = {ci_upper_spear - ci_lower_spear:.6f}")
            
            # Robustness assessment
            diff = pearson_r - spearman_rho
            print(f"\n  Δ (Pearson - Spearman) = {diff:+.6f}")
            if abs(diff) < 0.01:
                print(f"  ✓ Excellent agreement: Linear, minimal outliers")
            elif abs(diff) < 0.02:
                print(f"  ✓ Good agreement: Mostly linear")
            else:
                print(f"  ⚠ Check for outliers or non-linearity")
    
    # ================================================================
    # CREATE FIGURE
    # ================================================================
    if compute_spearman:
        # Two-panel figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        # Single panel (your original)
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = None
    
    x_pos = np.arange(len(model_names))
    colors = [MODEL_COLORS.get(name, '#1f77b4') for name in model_names]
    
    # ---- PANEL 1: Pearson (your original styling) ----
    ax1.bar(x_pos, pearson_rs, color=colors, alpha=0.8, 
            edgecolor='black', linewidth=1.2)
    
    errors = np.array([
        [pearson_rs[i] - pearson_ci_lowers[i] for i in range(len(model_names))],
        [pearson_ci_uppers[i] - pearson_rs[i] for i in range(len(model_names))]
    ])
    
    ax1.errorbar(x_pos, pearson_rs, yerr=errors, 
                fmt='none', ecolor='black', capsize=8, capthick=1.5, 
                linewidth=1.5, zorder=10)
    
    # Wheat boxes for Pearson
    for i, (model_name, r, ci_lower, ci_upper, p_val) in enumerate(zip(
            model_names, pearson_rs, pearson_ci_lowers, pearson_ci_uppers, p_values_pearson)):
        
        p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ci_width = ci_upper - ci_lower
        
        stats_text = (f"Pearson's R = {r:.4f}\n"
                      f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]\n"
                      f"CI width = {ci_width:.6f}\n"
                      f"{p_text}")
        
        ax1.text(i, 0.5 * ax1.get_ylim()[1], stats_text,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_ylabel("Pearson's R (Aggregate)", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    
    if not compute_spearman:
        ax1.set_title("Model Performance: Pearson's R with 95% CI\n(Fisher Z transformation)",
                     fontsize=13, fontweight='bold', pad=20)
    else:
        ax1.set_title(f"Pearson Correlation\n(N={n_samples:,}, Fisher Z CI)",
                     fontsize=13, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    
    y_max = max(pearson_ci_uppers) + 0.15
    ax1.set_ylim(0, min(1.0, y_max))
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # ---- PANEL 2: Spearman (if requested) ----
    if compute_spearman and ax2 is not None:
        ax2.bar(x_pos, spearman_rhos, color=colors, alpha=0.8,
                edgecolor='black', linewidth=1.2)
        
        errors_spear = np.array([
            [spearman_rhos[i] - spearman_ci_lowers[i] for i in range(len(model_names))],
            [spearman_ci_uppers[i] - spearman_rhos[i] for i in range(len(model_names))]
        ])
        
        ax2.errorbar(x_pos, spearman_rhos, yerr=errors_spear,
                    fmt='none', ecolor='black', capsize=8, capthick=1.5,
                    linewidth=1.5, zorder=10)
        
        # Wheat boxes for Spearman
        for i, (model_name, rho, ci_lower, ci_upper, p_val) in enumerate(zip(
                model_names, spearman_rhos, spearman_ci_lowers, spearman_ci_uppers, p_values_spearman)):
            
            p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            ci_width = ci_upper - ci_lower
            
            stats_text = (f"Spearman's ρ = {rho:.4f}\n"
                          f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]\n"
                          f"CI width = {ci_width:.6f}\n"
                          f"{p_text}")
            
            ax2.text(i, 0.5 * ax2.get_ylim()[1], stats_text,
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_ylabel("Spearman's ρ (Aggregate)", fontsize=12, fontweight='bold')
        ax2.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        ax2.set_title(f"Spearman Correlation\n(N={n_spear:,}, Fisher Z CI)",
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, fontsize=11, fontweight='bold')
        
        y_max_spear = max(spearman_ci_uppers) + 0.15
        ax2.set_ylim(0, min(1.0, y_max_spear))
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Figure saved to {output_path}")
    print(f"{'='*70}")
    plt.show()
    
    # ================================================================
    # RETURN METRICS
    # ================================================================
    metrics_summary = {}
    for i, model_name in enumerate(model_names):
        metrics_summary[model_name] = {
            'pearson_r': pearson_rs[i],
            'pearson_ci_lower': pearson_ci_lowers[i],
            'pearson_ci_upper': pearson_ci_uppers[i],
            'pearson_ci_width': pearson_ci_uppers[i] - pearson_ci_lowers[i],
            'pearson_p_value': p_values_pearson[i]
        }
        
        if compute_spearman:
            metrics_summary[model_name].update({
                'spearman_rho': spearman_rhos[i],
                'spearman_ci_lower': spearman_ci_lowers[i],
                'spearman_ci_upper': spearman_ci_uppers[i],
                'spearman_ci_width': spearman_ci_uppers[i] - spearman_ci_lowers[i],
                'spearman_p_value': p_values_spearman[i],
                'pearson_minus_spearman': pearson_rs[i] - spearman_rhos[i]
            })
    
    return metrics_summary


def figure_per_gene_dual_correlation_distribution(y_true_df, predictions_dict,
                                                  compute_spearman=True,
                                                  spearman_subsample_per_gene=None,
                                                  ci_method='analytical',
                                                  show_ci=True,
                                                  output_path='per_gene_dual_correlation.png'):
    """
    Generate violin plots showing distribution of per-gene Pearson AND Spearman correlations.
    
    This extends your existing per-gene analysis to include Spearman as a robustness check,
    using Fisher Z (analytical) CIs for both metrics.
    
    Parameters
    ----------
    y_true_df : pd.DataFrame, shape (n_samples, n_genes)
        True target values with gene names as columns
    predictions_dict : dict
        Model predictions {model_name: predictions_array}
    compute_spearman : bool, default=True
        Whether to compute Spearman alongside Pearson per gene
    spearman_subsample_per_gene : int, optional
        Subsample size for PER-GENE Spearman computation
        Default None means use all samples per gene (recommended for per-gene analysis)
        Only set this if you have >1M samples per gene and need speed
    ci_method : str, default='analytical'
        - 'analytical': Fast analytical CI using Fisher Z (RECOMMENDED)
        - 'bootstrap': Standard bootstrap (slow)
    show_ci : bool, default=True
        Whether to display confidence intervals
    output_path : str
        Where to save the figure
        
    Returns
    -------
    df_per_gene : pd.DataFrame
        Per-gene correlation values for each model
    summary_df : pd.DataFrame
        Summary statistics (median, mean, CIs) for each model
        
    Notes
    -----
    For per-gene analysis, we typically don't subsample since each gene's
    sample size is the number of cells/samples (e.g., 12,745), not 51M.
    Spearman computation is fast at this scale.
    """
    
    set_publication_style()
    
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    per_gene_data = []
    summary_stats = {}
    
    for model_name in model_names:
        y_pred = predictions_dict[model_name]
        n_genes = y_true_df.shape[1]
        n_samples_per_gene = y_true_df.shape[0]
        
        print(f"\n{'='*70}")
        print(f"Computing per-gene correlations for {model_name}")
        print(f"  {n_genes:,} genes × {n_samples_per_gene:,} samples per gene")
        print(f"{'='*70}")
        
        model_pearson_rs = []
        model_spearman_rhos = []
        
        # Compute per-gene correlations
        print(f"\nProcessing genes...")
        for gene_idx in range(n_genes):
            y_true_gene = y_true_df.iloc[:, gene_idx].values
            y_pred_gene = y_pred[:, gene_idx]
            
            # Only compute if gene has variance
            if np.var(y_true_gene) > 1e-10:
                # Pearson
                r_pearson, p_pearson = pearsonr(y_true_gene, y_pred_gene)
                
                per_gene_record = {
                    'model': model_name,
                    'gene': y_true_df.columns[gene_idx],
                    'pearson_r': r_pearson,
                    'pearson_p_value': p_pearson
                }
                
                model_pearson_rs.append(r_pearson)
                
                # Spearman (if requested)
                if compute_spearman:
                    # For per-gene, we typically don't subsample since
                    # n_samples_per_gene is usually small enough (<100k)
                    if spearman_subsample_per_gene and n_samples_per_gene > spearman_subsample_per_gene:
                        np.random.seed(42 + gene_idx)  # Reproducible per gene
                        subsample_idx = np.random.choice(n_samples_per_gene,
                                                        size=spearman_subsample_per_gene,
                                                        replace=False)
                        rho_spearman, p_spearman = spearmanr(y_true_gene[subsample_idx],
                                                             y_pred_gene[subsample_idx])
                    else:
                        rho_spearman, p_spearman = spearmanr(y_true_gene, y_pred_gene)
                    
                    per_gene_record['spearman_rho'] = rho_spearman
                    per_gene_record['spearman_p_value'] = p_spearman
                    per_gene_record['pearson_minus_spearman'] = r_pearson - rho_spearman
                    
                    model_spearman_rhos.append(rho_spearman)
                
                per_gene_data.append(per_gene_record)
            
            # Progress indicator
            if (gene_idx + 1) % 2000 == 0:
                print(f"  Processed {gene_idx + 1:,}/{n_genes:,} genes...")
        
        print(f"  ✓ Completed all {n_genes:,} genes")
        
        # ================================================================
        # COMPUTE SUMMARY STATISTICS
        # ================================================================
        model_pearson_rs = np.array(model_pearson_rs)
        median_r = np.median(model_pearson_rs)
        mean_r = np.mean(model_pearson_rs)
        
        summary_stats[model_name] = {
            'pearson_rs': model_pearson_rs,
            'pearson_median': median_r,
            'pearson_mean': mean_r,
            'pearson_q1': np.percentile(model_pearson_rs, 25),
            'pearson_q3': np.percentile(model_pearson_rs, 75)
        }
        
        if compute_spearman:
            model_spearman_rhos = np.array(model_spearman_rhos)
            median_rho = np.median(model_spearman_rhos)
            mean_rho = np.mean(model_spearman_rhos)
            
            summary_stats[model_name].update({
                'spearman_rhos': model_spearman_rhos,
                'spearman_median': median_rho,
                'spearman_mean': mean_rho,
                'spearman_q1': np.percentile(model_spearman_rhos, 25),
                'spearman_q3': np.percentile(model_spearman_rhos, 75),
                'median_difference': median_r - median_rho,
                'mean_difference': mean_r - mean_rho
            })
        
        # ================================================================
        # COMPUTE CONFIDENCE INTERVALS FOR SUMMARY STATS
        # ================================================================
        if show_ci and ci_method == 'analytical':
            print(f"\nComputing analytical CIs for {model_name}...")
            
            n_genes_valid = len(model_pearson_rs)
            
            # Pearson CIs
            se_mean = np.std(model_pearson_rs) / np.sqrt(n_genes_valid)
            summary_stats[model_name]['pearson_mean_ci_lower'] = mean_r - 1.96 * se_mean
            summary_stats[model_name]['pearson_mean_ci_upper'] = mean_r + 1.96 * se_mean
            
            se_median = 1.253 * se_mean  # Asymptotic relationship
            summary_stats[model_name]['pearson_median_ci_lower'] = median_r - 1.96 * se_median
            summary_stats[model_name]['pearson_median_ci_upper'] = median_r + 1.96 * se_median
            
            print(f"  Pearson Median: {median_r:.4f} "
                  f"[{summary_stats[model_name]['pearson_median_ci_lower']:.4f}, "
                  f"{summary_stats[model_name]['pearson_median_ci_upper']:.4f}]")
            
            # Spearman CIs (if computed)
            if compute_spearman:
                se_mean_spear = np.std(model_spearman_rhos) / np.sqrt(n_genes_valid)
                summary_stats[model_name]['spearman_mean_ci_lower'] = mean_rho - 1.96 * se_mean_spear
                summary_stats[model_name]['spearman_mean_ci_upper'] = mean_rho + 1.96 * se_mean_spear
                
                se_median_spear = 1.253 * se_mean_spear
                summary_stats[model_name]['spearman_median_ci_lower'] = median_rho - 1.96 * se_median_spear
                summary_stats[model_name]['spearman_median_ci_upper'] = median_rho + 1.96 * se_median_spear
                
                print(f"  Spearman Median: {median_rho:.4f} "
                      f"[{summary_stats[model_name]['spearman_median_ci_lower']:.4f}, "
                      f"{summary_stats[model_name]['spearman_median_ci_upper']:.4f}]")
        
        elif show_ci and ci_method == 'bootstrap':
            print(f"\nComputing bootstrap CIs for {model_name} (this may take a while)...")
            # Standard bootstrap implementation (kept for compatibility)
            n_bootstrap = 1000
            n_genes_valid = len(model_pearson_rs)
            
            pearson_median_boots = []
            pearson_mean_boots = []
            
            if compute_spearman:
                spearman_median_boots = []
                spearman_mean_boots = []
            
            np.random.seed(42)
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(n_genes_valid, size=n_genes_valid, replace=True)
                
                boot_pearson = model_pearson_rs[boot_indices]
                pearson_median_boots.append(np.median(boot_pearson))
                pearson_mean_boots.append(np.mean(boot_pearson))
                
                if compute_spearman:
                    boot_spearman = model_spearman_rhos[boot_indices]
                    spearman_median_boots.append(np.median(boot_spearman))
                    spearman_mean_boots.append(np.mean(boot_spearman))
            
            summary_stats[model_name]['pearson_median_ci_lower'] = np.percentile(pearson_median_boots, 2.5)
            summary_stats[model_name]['pearson_median_ci_upper'] = np.percentile(pearson_median_boots, 97.5)
            summary_stats[model_name]['pearson_mean_ci_lower'] = np.percentile(pearson_mean_boots, 2.5)
            summary_stats[model_name]['pearson_mean_ci_upper'] = np.percentile(pearson_mean_boots, 97.5)
            
            if compute_spearman:
                summary_stats[model_name]['spearman_median_ci_lower'] = np.percentile(spearman_median_boots, 2.5)
                summary_stats[model_name]['spearman_median_ci_upper'] = np.percentile(spearman_median_boots, 97.5)
                summary_stats[model_name]['spearman_mean_ci_lower'] = np.percentile(spearman_mean_boots, 2.5)
                summary_stats[model_name]['spearman_mean_ci_upper'] = np.percentile(spearman_mean_boots, 97.5)
    
    # Convert to DataFrame
    df_per_gene = pd.DataFrame(per_gene_data)
    
    # ================================================================
    # CREATE FIGURE
    # ================================================================
    if compute_spearman:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = None
    
    # ---- PANEL 1: Pearson Distribution ----
    parts1 = ax1.violinplot(
        [df_per_gene[df_per_gene['model'] == m]['pearson_r'].values 
         for m in model_names],
        positions=np.arange(n_models),
        widths=0.7,
        showmeans=True,
        showmedians=True
    )
    
    for i, pc in enumerate(parts1['bodies']):
        pc.set_facecolor(MODEL_COLORS.get(model_names[i], '#1f77b4'))
        pc.set_alpha(0.6)
    
    # Add CI error bars for Pearson median
    if show_ci and ci_method in ['analytical', 'bootstrap']:
        for i, model_name in enumerate(model_names):
            stats = summary_stats[model_name]
            median = stats['pearson_median']
            
            if 'pearson_median_ci_lower' in stats:
                ci_lower = stats['pearson_median_ci_lower']
                ci_upper = stats['pearson_median_ci_upper']
                
                ax1.errorbar(i, median,
                           yerr=[[median - ci_lower], [ci_upper - median]],
                           fmt='D', color='darkred', markersize=6,
                           capsize=10, capthick=2, linewidth=2,
                           label='Median 95% CI' if i == 0 else '', zorder=10)
    
    # Wheat boxes for Pearson
    for i, model_name in enumerate(model_names):
        stats = summary_stats[model_name]
        median = stats['pearson_median']
        mean = stats['pearson_mean']
        q1, q3 = stats['pearson_q1'], stats['pearson_q3']
        
        if show_ci and 'pearson_median_ci_lower' in stats:
            textstr = (f"Median = {median:.3f}\n"
                      f"95% CI = [{stats['pearson_median_ci_lower']:.3f}, "
                      f"{stats['pearson_median_ci_upper']:.3f}]\n"
                      f"Mean = {mean:.3f}\n"
                      f"IQR = [{q1:.3f}, {q3:.3f}]")
        else:
            textstr = f'Median = {median:.3f}\nMean = {mean:.3f}\nIQR = [{q1:.3f}, {q3:.3f}]'
        
        ax1.text(i, ax1.get_ylim()[1] * 0.5, textstr,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_ylabel("Pearson's R (Per-Gene)", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    
    n_genes_total = len(df_per_gene[df_per_gene['model'] == model_names[0]])
    if not compute_spearman:
        ax1.set_title(f"Distribution of Per-Gene Pearson's R ({n_genes_total:,} genes)\n"
                     f"95% CI via Fisher Z (analytical)",
                     fontsize=13, fontweight='bold', pad=20)
    else:
        ax1.set_title(f"Per-Gene Pearson Correlation ({n_genes_total:,} genes)",
                     fontsize=13, fontweight='bold')
    
    ax1.set_xticks(np.arange(n_models))
    ax1.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    if show_ci:
        ax1.legend(loc='lower right', fontsize=9)
    
    # ---- PANEL 2: Spearman Distribution (if computed) ----
    if compute_spearman and ax2 is not None:
        parts2 = ax2.violinplot(
            [df_per_gene[df_per_gene['model'] == m]['spearman_rho'].values 
             for m in model_names],
            positions=np.arange(n_models),
            widths=0.7,
            showmeans=True,
            showmedians=True
        )
        
        for i, pc in enumerate(parts2['bodies']):
            pc.set_facecolor(MODEL_COLORS.get(model_names[i], '#1f77b4'))
            pc.set_alpha(0.6)
        
        # Add CI error bars for Spearman median
        if show_ci and ci_method in ['analytical', 'bootstrap']:
            for i, model_name in enumerate(model_names):
                stats = summary_stats[model_name]
                median = stats['spearman_median']
                
                if 'spearman_median_ci_lower' in stats:
                    ci_lower = stats['spearman_median_ci_lower']
                    ci_upper = stats['spearman_median_ci_upper']
                    
                    ax2.errorbar(i, median,
                               yerr=[[median - ci_lower], [ci_upper - median]],
                               fmt='D', color='darkred', markersize=6,
                               capsize=10, capthick=2, linewidth=2,
                               label='Median 95% CI' if i == 0 else '', zorder=10)
        
        # Wheat boxes for Spearman
        for i, model_name in enumerate(model_names):
            stats = summary_stats[model_name]
            median = stats['spearman_median']
            mean = stats['spearman_mean']
            q1, q3 = stats['spearman_q1'], stats['spearman_q3']
            
            if show_ci and 'spearman_median_ci_lower' in stats:
                textstr = (f"Median = {median:.3f}\n"
                          f"95% CI = [{stats['spearman_median_ci_lower']:.3f}, "
                          f"{stats['spearman_median_ci_upper']:.3f}]\n"
                          f"Mean = {mean:.3f}\n"
                          f"IQR = [{q1:.3f}, {q3:.3f}]")
            else:
                textstr = f'Median = {median:.3f}\nMean = {mean:.3f}\nIQR = [{q1:.3f}, {q3:.3f}]'
            
            ax2.text(i, ax2.get_ylim()[1] * 0.5, textstr,
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_ylabel("Spearman's ρ (Per-Gene)", fontsize=12, fontweight='bold')
        ax2.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        ax2.set_title(f"Per-Gene Spearman Correlation ({n_genes_total:,} genes)",
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(np.arange(n_models))
        ax2.set_xticklabels(model_names, fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        if show_ci:
            ax2.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Figure saved to {output_path}")
    print(f"{'='*70}")
    plt.show()
    
    # ================================================================
    # SUMMARY OUTPUT
    # ================================================================
    summary_df = pd.DataFrame(summary_stats).T
    
    print(f"\n{'='*70}")
    print("PER-GENE SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    for model_name in model_names:
        stats = summary_stats[model_name]
        print(f"\n{model_name}:")
        print(f"  Pearson - Median: {stats['pearson_median']:.4f}, Mean: {stats['pearson_mean']:.4f}")
        
        if compute_spearman:
            print(f"  Spearman - Median: {stats['spearman_median']:.4f}, Mean: {stats['spearman_mean']:.4f}")
            print(f"  Difference (Pearson - Spearman):")
            print(f"    Median: {stats['median_difference']:+.4f}")
            print(f"    Mean: {stats['mean_difference']:+.4f}")
            
            if abs(stats['median_difference']) < 0.01:
                print(f"    ✓ Excellent per-gene agreement")
            elif abs(stats['median_difference']) < 0.02:
                print(f"    ✓ Good per-gene agreement")
            else:
                print(f"    ⚠ Notable per-gene differences detected")
    
    print(f"{'='*70}")
    
    return df_per_gene, summary_df
