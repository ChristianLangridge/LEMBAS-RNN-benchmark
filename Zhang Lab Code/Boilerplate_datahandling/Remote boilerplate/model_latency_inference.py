import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
import time
from typing import Dict, List, Tuple
import warnings
import sys

# Add your specific paths here if not already in environment
# sys.path.append('/path/to/your/modules')

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration and Styling
# ============================================================================

MODEL_COLORS = {
    'RNN': '#1f77b4',           # Deep blue
    'XGBRFRegressor': '#ff7f0e', # Orange
    'MLR': '#2ca02c'            # Green
}

def set_publication_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.dpi': 300,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

# ============================================================================
# Revised Benchmarking Functions (Latency Focused)
# ============================================================================

def benchmark_mlr_inference(model, X_input: np.ndarray, n_runs: int = 100, 
                           warmup_runs: int = 20) -> Dict[str, float]:
    """Benchmarks MLR on the specific input provided (single row or batch)."""
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(X_input)
    
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_input)
        latencies.append((time.perf_counter() - start) * 1000) # ms
        
    return _calculate_stats(latencies)


def benchmark_xgbrf_inference(models_list: List, X_input: np.ndarray, 
                              n_runs: int = 100, warmup_runs: int = 20) -> Dict[str, float]:
    """Benchmarks the list of XGB models on the input."""
    
    # Warmup
    for _ in range(warmup_runs):
        _ = [model.predict(X_input) for model in models_list]
    
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        # Note: The list comprehension is part of the inference cost here
        # For pure prediction speed, we exclude the column_stack overhead 
        # unless you specifically need the formatted array output.
        _ = [model.predict(X_input) for model in models_list]
        latencies.append((time.perf_counter() - start) * 1000) # ms
    
    return _calculate_stats(latencies)


def benchmark_rnn_inference(model, X_input: torch.Tensor, n_runs: int = 100, 
                           warmup_runs: int = 20) -> Dict[str, float]:
    """Benchmarks RNN on the specific input tensor."""
    
    # Ensure model is in eval mode
    model.eval()
   
    # DEBUG: Check what the model thinks it's doing
    print(f"DEBUGGING: Input shape: {X_input.shape}")
    try:
        # For LSTM/GRU, weight_ih_l0 size reveals the expected input feature size
        print(f"DEBUG: Model expects input_size: {model.rnn.input_size}")
    except:
        pass
        
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(X_input)
    
    latencies = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(X_input)
            latencies.append((time.perf_counter() - start) * 1000) # ms
            
    return _calculate_stats(latencies)


def _calculate_stats(latencies_list):
    arr = np.array(latencies_list)
    return {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr),
        'p95': np.percentile(arr, 95),
        'p99': np.percentile(arr, 99),
        'all_latencies': arr
    }

# ============================================================================
# Main Execution Logic
# ============================================================================

def run_benchmarks(mlr_model, xgbrf_models, rnn_model, X_full, 
                   n_runs=100, save_path=None):
    
    print("\n" + "="*80)
    print(" 🚀 STARTING DUAL-MODE BENCHMARK")
    print("="*80)

    # ---------------------------------------------------------
    # 1. PREPARE INPUTS
    # ---------------------------------------------------------
    # Helper to get single row
    if isinstance(X_full, pd.DataFrame):
        X_np_full = X_full.values
        # Create single row (1, n_features) to preserve 2D shape
        X_np_single = X_full.iloc[0:1].values 
    else:
        X_np_full = X_full
        X_np_single = X_full[0:1, :]

    # Prepare Tensors for RNN
    # We assume the RNN model was trained on the same device (CPU usually for inference tests)
    # Prepare Tensors for RNN
    X_torch_full = torch.tensor(X_np_full, dtype=torch.float32)
    X_torch_single = torch.tensor(X_np_single, dtype=torch.float32)
 
    # ---------------------------------------------------------
    # 2. RUN TRUE LATENCY TEST (Batch Size = 1)
    # ---------------------------------------------------------
    print(f"\n[Mode A] TRUE LATENCY (Input shape: {X_np_single.shape})")
    print("-" * 60)
    
    results_latency = {}
    
    print(f"  Testing MLR...")
    results_latency['MLR'] = benchmark_mlr_inference(mlr_model, X_np_single, n_runs)
    
    print(f"  Testing XGBRFRegressor ({len(xgbrf_models)} sub-models)...")
    results_latency['XGBRFRegressor'] = benchmark_xgbrf_inference(xgbrf_models, X_np_single, n_runs)
    
    print(f"  Testing RNN...")
    # Try/Except block in case RNN is strictly graph-based and rejects single nodes
    try:
        results_latency['RNN'] = benchmark_rnn_inference(rnn_model, X_torch_single, n_runs)
    except Exception as e:
        print(f"    ! RNN Single-Row Error: {e}")
        print("    ! Falling back to full batch for RNN (Latency will be inflated)")
        results_latency['RNN'] = benchmark_rnn_inference(rnn_model, X_torch_full, n_runs)

    # ---------------------------------------------------------
    # 3. RUN THROUGHPUT TEST (Batch Size = Full)
    # ---------------------------------------------------------
    print(f"\n[Mode B] BATCH THROUGHPUT (Input shape: {X_np_full.shape})")
    print("-" * 60)
    
    # We run fewer runs for batch processing as it takes longer
    results_throughput = {}
    results_throughput['MLR'] = benchmark_mlr_inference(mlr_model, X_np_full, n_runs=10)['mean']
    results_throughput['XGBRFRegressor'] = benchmark_xgbrf_inference(xgbrf_models, X_np_full, n_runs=5)['mean']
    results_throughput['RNN'] = benchmark_rnn_inference(rnn_model, X_torch_full, n_runs=10)['mean']

    # ---------------------------------------------------------
    # 4. REPORTING
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<15} | {'Latency (ms/sample)':<22} | {'Throughput (samples/sec)':<25}")
    print("-" * 70)
    
    batch_size = X_np_full.shape[0]
    
    for model in ['MLR', 'XGBRFRegressor', 'RNN']:
        lat = results_latency[model]['mean']
        # Throughput = (Batch Size / Batch Time in seconds)
        batch_time_ms = results_throughput[model]
        tput = batch_size / (batch_time_ms / 1000)
        
        print(f"{model:<15} | {lat:>10.3f} ms           | {tput:>10.1f} samples/s")

    # ---------------------------------------------------------
    # 5. PLOTTING (Only plotting Latency as requested)
    # ---------------------------------------------------------
    plot_latency_comparison(results_latency, save_path)
    
    return results_latency

# ============================================================================
# Plotting Function (Unchanged, just ensured it uses the results)
# ============================================================================

def plot_latency_comparison(results, save_path):
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_names = list(results.keys())
    
    # 1. Box Plot
    ax = axes[0]
    positions = [1, 2, 3]
    for i, model in enumerate(model_names):
        data = results[model]['all_latencies']
        color = MODEL_COLORS.get(model, 'gray')
        ax.boxplot([data], positions=[positions[i]], patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.5),
                   medianprops=dict(color='black'))
        # Jitter
        y = data
        x = np.random.normal(positions[i], 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=10, color=color)
        
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names)
    ax.set_title("Distribution (Single Row)")
    ax.set_ylabel("Latency (ms)")

    # 2. Bar Mean
    ax = axes[1]
    means = [results[m]['mean'] for m in model_names]
    stds = [results[m]['std'] for m in model_names]
    ax.bar(model_names, means, yerr=stds, capsize=5, 
           color=[MODEL_COLORS.get(m) for m in model_names], alpha=0.7)
    for i, v in enumerate(means):
        ax.text(i, v + stds[i], f"{v:.2f} ms", ha='center', va='bottom', fontweight='bold')
    ax.set_title("Mean Latency (Single Row)")

    # 3. Percentiles
    ax = axes[2]
    x = np.arange(len(model_names))
    width = 0.25
    for i, metric in enumerate(['median', 'p99']):
        vals = [results[m][metric] for m in model_names]
        ax.bar(x + (i*width), vals, width, label=metric, alpha=0.8)
    
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_title("Median vs P99 Tail Latency")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"\n✓ Saved figure to {save_path}")
    plt.show()
