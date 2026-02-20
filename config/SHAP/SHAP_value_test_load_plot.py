import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from dataclasses import dataclass, field
from typing import Dict, List
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

# --- 1. Your Defined Constants & Style ---
MODEL_COLORS = {
    'RNN': '#1f77b4',            # Deep blue
    'XGBRFRegressor': '#ff7f0e', # Orange
    'MLR': '#2ca02c'             # Green
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
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
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

# --- 2. Data Structures ---
@dataclass
class GeneShapData:
    shap_values: np.ndarray
    base_value: float
    
@dataclass
class ModelData:
    name: str
    feature_names: List[str]
    genes: Dict[str, GeneShapData] = field(default_factory=dict)

class ShapAnalysisPipeline:
    def __init__(self, baseline_path: str, rnn_path: str):
        self.models: Dict[str, ModelData] = {}
        self._load_data(baseline_path, rnn_path)

    def _load_data(self, baseline_path: str, rnn_path: str):
        print(f"--- Loading SHAP Data ---")
        try:
            # Load Baselines
            b_data = np.load(baseline_path, allow_pickle=True)
            feature_names = b_data['feature_names']
            
            # MLR
            mlr = ModelData('MLR', feature_names)
            mlr.genes['ALB'] = GeneShapData(b_data['MLR_ALB_shap_values'], b_data['MLR_ALB_expected_value'])
            mlr.genes['AFP'] = GeneShapData(b_data['MLR_AFP_shap_values'], b_data['MLR_AFP_expected_value'])
            self.models['MLR'] = mlr
            
            # XGBRF
            xgb = ModelData('XGBRFRegressor', feature_names)
            xgb.genes['ALB'] = GeneShapData(b_data['XGBRF_ALB_shap_values'], b_data['XGBRF_ALB_expected_value'])
            xgb.genes['AFP'] = GeneShapData(b_data['XGBRF_AFP_shap_values'], b_data['XGBRF_AFP_expected_value'])
            self.models['XGBRFRegressor'] = xgb
            
            # Load RNN
            r_data = np.load(rnn_path, allow_pickle=True)
            rnn = ModelData('RNN', r_data['feature_names'])
            rnn.genes['ALB'] = GeneShapData(r_data['RNN_ALB_shap_values'], r_data['RNN_ALB_expected_value'])
            rnn.genes['AFP'] = GeneShapData(r_data['RNN_AFP_shap_values'], r_data['RNN_AFP_expected_value'])
            self.models['RNN'] = rnn
            
            print(f"✅ Loaded Models: {list(self.models.keys())}")
        except KeyError as e:
            print(f"❌ Error loading data: {e}")
            raise

    def plot_benchmark_grid(self, output_file="shap_benchmark_poster.png"):
        """
        Generates a 2x3 grid optimized for an A1 Poster.
        Uses your base style, but overrides sizes for distance visibility.
        """
        # 1. Apply your Base Style
        set_publication_style()
        
        # 2. Apply Poster Overrides (Scale up for A1 visibility)
        # Note: We must scale up fonts/sizes, otherwise they are unreadable from 2m.
        plt.rcParams.update({
            'font.size': 11,              
            'axes.labelsize': 12,         
            'axes.titlesize': 13,         
            'xtick.labelsize': 10,        
            'ytick.labelsize': 10,        
            # Keep your requested linewidths proportional or slightly thicker for poster
            'axes.linewidth': 1.2,        
            'xtick.major.width': 5,
            'ytick.major.width': 5,
        })

        genes = ['ALB', 'AFP']
        model_order = ['MLR', 'XGBRFRegressor', 'RNN']
        
        # 3. Canvas Size: Needs to be much larger than FIGSIZE_WIDE to allow spacing
        # We use (34, 22) to prevent overlap while keeping elements large.
        fig, axes = plt.subplots(len(genes), len(model_order), figsize=(75, 50))
        
        for row, gene_name in enumerate(genes):
            # Find Best Sample
            mlr_data = self.models['MLR'].genes[gene_name]
            preds = np.sum(mlr_data.shap_values, axis=1) + mlr_data.base_value
            sample_idx = np.argmax(preds)
            print(f"Plotting {gene_name} Sample #{sample_idx}")
            
            for col, model_name in enumerate(model_order):
                ax = axes[row, col]
                model = self.models[model_name]
                gene_data = model.genes[gene_name]
                
                # Data Prep
                vals = gene_data.shap_values[sample_idx]
                if vals.ndim > 1: vals = vals.flatten()
                base = float(gene_data.base_value)

                expl = shap.Explanation(
                    values=vals,
                    base_values=base,
                    data=None, 
                    feature_names=model.feature_names
                )
                
                # --- Plotting ---
                plt.sca(ax)
                shap.plots.waterfall(expl, max_display=9, show=False)
                
                # Store references to the SHAP-generated E[f(X)] and f(x) texts before removing them
                e_fx_text = None
                f_x_text = None
                
                # Collect and remove SHAP-generated special labels
                texts_to_remove = []
                for txt in ax.texts:
                    text_content = txt.get_text()
                    if "E[f(X)]" in text_content:
                        e_fx_value = text_content  # Store the full text
                        texts_to_remove.append(txt)
                    elif "f(x)" in text_content.lower() and "e[f(x)]" not in text_content.lower():
                        f_x_value = text_content  # Store the full text
                        texts_to_remove.append(txt)
                
                # Remove the SHAP-generated texts
                for txt in texts_to_remove:
                    txt.remove()
                
                # Modify label sizes for remaining feature contributions
                for txt in ax.texts:
                    text_content = txt.get_text()
                    if all(x not in text_content for x in ["E[f(X)]", "f(X)", "f(x)", "other"]):
                        txt.set_fontsize(10)
                
                # ---------------------------------------------------------
                # Manually add aligned E[f(X)] and f(x) labels with strict positioning
                # ---------------------------------------------------------
                
                # Get axis limits to position labels properly
                ax_xlim = ax.get_xlim()
                ax_ylim = ax.get_ylim()
                
                # Add E[f(X)] - centered at bottom
                if e_fx_text:
                    ax.text(0.5, -0.12, e_fx_value, 
                            transform=ax.transAxes,
                            horizontalalignment='center',
                            verticalalignment='top',
                            fontsize=11,
                            fontweight='bold')
                
                # Add f(x) - centered at top
                if f_x_text:
                    ax.text(0.5, 1.05, f_x_value,
                            transform=ax.transAxes,
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            fontsize=11,
                            fontweight='bold')
                
                # Modify label sizes
                for txt in ax.texts:
                    text_content = txt.get_text()
                    # Skip the special labels (E[f(X)], f(X), etc.)
                    if all(x not in text_content for x in ["E[f(X)]", "f(X)", "other"]):
                        txt.set_fontsize(10)

                # --- Custom Aesthetics ---
                # Title Format: "MLR (ALB)"
                title_str = f"{model_name} ({gene_name})"
                ax.set_title(title_str, 
                             color=MODEL_COLORS[model_name], 
                             fontsize=13, 
                             fontweight='bold', 
                             pad=40) 
                
                # X-Label: Pushed down to clear E[f(x)]
                ax.set_xlabel("SHAP Value (Impact)", 
                              fontsize=10, 
                              fontweight='bold', 
                              labelpad=20) 
                
                # Feature Names (Y-axis)
                ax.tick_params(axis='y', labelsize=10) 
                
                # Boxed Spines (Matching your style)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#333333')
                    spine.set_linewidth(1.2) 
                    
                # Grid
                ax.grid(True, which='major', axis='x', color='#EEEEEE', linestyle='-', linewidth=1.5)
                ax.set_axisbelow(True)

        # --- 4. Spacing Strategy ---
        # wspace=1.0: Full plot width gap between columns
        # hspace=1.2: 120% plot height gap between ALB and AFP rows
        plt.subplots_adjust(
            wspace=1.2,  
            hspace=1.2,  
            left=0.08, right=0.95, top=0.92, bottom=0.08
        )
        
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"\n✅ Poster Figure saved to: {os.path.abspath(output_file)}")