import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from dataclasses import dataclass, field
from typing import Dict, List

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
            'font.size': 16,              
            'axes.labelsize': 20,         
            'axes.titlesize': 24,         
            'xtick.labelsize': 16,        
            'ytick.labelsize': 16,        
            # Keep your requested linewidths proportional or slightly thicker for poster
            'axes.linewidth': 2.0,        
            'xtick.major.width': 2.0,
            'ytick.major.width': 2.0,
        })

        genes = ['ALB', 'AFP']
        model_order = ['MLR', 'XGBRFRegressor', 'RNN']
        
        # 3. Canvas Size: Needs to be much larger than FIGSIZE_WIDE to allow spacing
        # We use (34, 22) to prevent overlap while keeping elements large.
        fig, axes = plt.subplots(len(genes), len(model_order), figsize=(34, 22))
        
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
                
                # --- Custom Aesthetics ---
                # Title Format: "MLR (ALB)"
                title_str = f"{model_name} ({gene_name})"
                ax.set_title(title_str, 
                             color=MODEL_COLORS[model_name], 
                             fontsize=24, 
                             fontweight='bold', 
                             pad=40) 
                
                # X-Label: Pushed down to clear E[f(x)]
                ax.set_xlabel("SHAP Value (Impact)", 
                              fontsize=20, 
                              fontweight='bold', 
                              labelpad=30) 
                
                # Feature Names (Y-axis)
                ax.tick_params(axis='y', labelsize=16) 
                
                # Boxed Spines (Matching your style)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#333333')
                    spine.set_linewidth(2.0) 
                    
                # Grid
                ax.grid(True, which='major', axis='x', color='#EEEEEE', linestyle='-', linewidth=1.5)
                ax.set_axisbelow(True)

        # --- 4. Spacing Strategy ---
        # wspace=1.0: Full plot width gap between columns
        # hspace=1.2: 120% plot height gap between ALB and AFP rows
        plt.subplots_adjust(
            wspace=1.0,  
            hspace=1.2,  
            left=0.08, right=0.95, top=0.92, bottom=0.08
        )
        
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"\n✅ Poster Figure saved to: {os.path.abspath(output_file)}")
