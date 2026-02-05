import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
import time
from typing import Dict, List, Tuple
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
# Latency benchmarking functions
# ============================================================================

def benchmark_mlr_inference(model, X_data: np.ndarray, n_runs: int = 100, 
                           warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark MLR model inference latency.
    
    Parameters:
    -----------
    model : sklearn model
        Trained MLR model
    X_data : np.ndarray
        Input features for inference
    n_runs : int
        Number of timing runs (default: 100)
    warmup_runs : int
        Number of warmup runs to exclude (default: 10)
    
    Returns:
    --------
    dict : Latency statistics in milliseconds
    """
    latencies = []
    
    # Warmup runs
    for _ in range(warmup_runs):
        _ = model.predict(X_data)
    
    # Timed runs
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'all_latencies': latencies
    }


def benchmark_xgbrf_inference(models_list: List, X_data: np.ndarray, 
                              n_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark XGBRF model inference latency.
    
    Parameters:
    -----------
    models_list : list
        List of trained XGBRF models (one per gene)
    X_data : np.ndarray
        Input features for inference
    n_runs : int
        Number of timing runs (default: 100)
    warmup_runs : int
        Number of warmup runs to exclude (default: 10)
    
    Returns:
    --------
    dict : Latency statistics in milliseconds
    """
    latencies = []
    
    # Warmup runs
    for _ in range(warmup_runs):
        _ = np.column_stack([model.predict(X_data) for model in models_list])
    
    # Timed runs
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = np.column_stack([model.predict(X_data) for model in models_list])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'all_latencies': latencies
    }


def benchmark_rnn_inference(model, n_runs: int = 100, 
                           warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark RNN model inference latency.
    
    Parameters:
    -----------
    model : torch model
        Trained RNN model with X_in already loaded
    n_runs : int
        Number of timing runs (default: 100)
    warmup_runs : int
        Number of warmup runs to exclude (default: 10)
    
    Returns:
    --------
    dict : Latency statistics in milliseconds
    """
    latencies = []
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(model.X_in)
    
    # Timed runs
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(model.X_in)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'all_latencies': latencies
    }


# ============================================================================
# Visualization functions
# ============================================================================

def plot_latency_comparison(results: Dict[str, Dict[str, float]], 
                           save_path: str = None,
                           show_percentiles: bool = True) -> plt.Figure:
    """
    Create publication-quality latency comparison figure.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model names as keys and latency stats as values
    save_path : str, optional
        Path to save the figure
    show_percentiles : bool
        Whether to show p95/p99 markers (default: True)
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRIPLE)
    
    model_names = list(results.keys())
    
    # ========================================================================
    # Panel A: Box plot with individual points
    # ========================================================================
    ax = axes[0]
    
    positions = []
    for i, model_name in enumerate(model_names):
        pos = i + 1
        positions.append(pos)
        latencies = results[model_name]['all_latencies']
        color = MODEL_COLORS[model_name]
        
        # Box plot
        bp = ax.boxplot([latencies], positions=[pos], widths=0.5,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.3, linewidth=1.2),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        
        # Overlay individual points with jitter
        jitter = np.random.normal(0, 0.04, size=len(latencies))
        ax.scatter(np.ones(len(latencies)) * pos + jitter, latencies, 
                  alpha=0.3, s=10, color=color, zorder=3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=0)
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('A. Latency Distribution', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Panel B: Bar chart with error bars (mean ± std)
    # ========================================================================
    ax = axes[1]
    
    means = [results[model]['mean'] for model in model_names]
    stds = [results[model]['std'] for model in model_names]
    colors = [MODEL_COLORS[model] for model in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=0)
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('B. Mean Latency ± SD', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Panel C: Percentile comparison (p50, p95, p99)
    # ========================================================================
    ax = axes[2]
    
    percentiles = ['median', 'p95', 'p99']
    percentile_labels = ['p50 (Median)', 'p95', 'p99']
    x = np.arange(len(percentiles))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        values = [results[model_name][p] for p in percentiles]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                     color=MODEL_COLORS[model_name], alpha=0.7, 
                     edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xticks(x)
    ax.set_xticklabels(percentile_labels)
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('C. Latency Percentiles', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig


def print_latency_summary(results: Dict[str, Dict[str, float]], 
                         batch_size: int = None):
    """
    Print formatted latency benchmark summary table.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model names as keys and latency stats as values
    batch_size : int, optional
        Batch size used for benchmarking
    """
    print("\n" + "="*85)
    print("INFERENCE LATENCY BENCHMARK SUMMARY")
    print("="*85)
    
    if batch_size:
        print(f"Batch size: {batch_size} samples")
    
    print("\n{:<18} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Mean (ms)", "Median", "Std Dev", "p95", "p99"))
    print("-"*85)
    
    for model_name in results.keys():
        stats = results[model_name]
        print("{:<18} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
            model_name,
            stats['mean'],
            stats['median'],
            stats['std'],
            stats['p95'],
            stats['p99']
        ))
    
    print("="*85)
    
    # Calculate speedup factors relative to slowest model
    mean_latencies = {model: results[model]['mean'] for model in results.keys()}
    slowest_model = max(mean_latencies, key=mean_latencies.get)
    slowest_latency = mean_latencies[slowest_model]
    
    print("\nSpeedup factors (relative to slowest model):")
    print("-"*85)
    for model_name in results.keys():
        speedup = slowest_latency / mean_latencies[model_name]
        print(f"{model_name:<18} {speedup:>6.2f}x {'(baseline)' if model_name == slowest_model else ''}")
    
    print("="*85 + "\n")
    
    # Throughput estimation
    if batch_size:
        print("Estimated throughput (samples/second):")
        print("-"*85)
        for model_name in results.keys():
            throughput = (batch_size / results[model_name]['mean']) * 1000
            print(f"{model_name:<18} {throughput:>10.1f} samples/s")
        print("="*85 + "\n")


# ============================================================================
# Main benchmarking workflow
# ============================================================================

def benchmark_all_models(mlr_model, xgbrf_models, rnn_model, X_test,
                        n_runs: int = 100, warmup_runs: int = 10,
                        save_path: str = None) -> Dict[str, Dict[str, float]]:
    """
    Benchmark all three models and create comparison figure.
    
    Parameters:
    -----------
    mlr_model : sklearn model
        Trained MLR model
    xgbrf_models : list
        List of trained XGBRF models
    rnn_model : torch model
        Trained RNN model
    X_test : np.ndarray or pd.DataFrame
        Test input features
    n_runs : int
        Number of timing runs per model (default: 100)
    warmup_runs : int
        Number of warmup runs (default: 10)
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    dict : Complete latency results for all models
    """
    print("\n" + "="*85)
    print("STARTING INFERENCE LATENCY BENCHMARK")
    print("="*85)
    print(f"Batch size: {X_test.shape[0]} samples")
    print(f"Number of runs per model: {n_runs}")
    print(f"Warmup runs: {warmup_runs}")
    print("="*85 + "\n")
    
    results = {}
    
    # Convert to numpy if DataFrame
    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values
    else:
        X_test_np = X_test
    
    # Benchmark MLR
    print("[1/3] Benchmarking MLR model...")
    results['MLR'] = benchmark_mlr_inference(
        mlr_model, X_test_np, n_runs=n_runs, warmup_runs=warmup_runs
    )
    print(f"      Mean latency: {results['MLR']['mean']:.3f} ms")
    
    # Benchmark XGBRF
    print("[2/3] Benchmarking XGBRFRegressor models...")
    results['XGBRFRegressor'] = benchmark_xgbrf_inference(
        xgbrf_models, X_test_np, n_runs=n_runs, warmup_runs=warmup_runs
    )
    print(f"      Mean latency: {results['XGBRFRegressor']['mean']:.3f} ms")
    
    # Benchmark RNN
    print("[3/3] Benchmarking RNN model...")
    results['RNN'] = benchmark_rnn_inference(
        rnn_model, n_runs=n_runs, warmup_runs=warmup_runs
    )
    print(f"      Mean latency: {results['RNN']['mean']:.3f} ms")
    
    print("\n✓ Benchmarking complete!\n")
    
    # Print summary table
    print_latency_summary(results, batch_size=X_test.shape[0])
    
    # Create visualization
    print("Generating latency comparison figure...")
    fig = plot_latency_comparison(results, save_path=save_path)
    
    return results


