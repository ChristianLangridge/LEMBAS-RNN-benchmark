"""
SHAP Feature Contribution Analysis for Model Comparison
Comparing MLR, XGBRF, and LEMBAS-RNN models

Author: Bioinformatics PhD Student
Purpose: Generate comparative SHAP visualizations across different model architectures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class SHAPModelComparator:
    """
    A class to compute and compare SHAP values across different model types.
    """
    
    def __init__(self, models_dict, X_data, feature_names=None, background_samples=100):
        """
        Initialize the comparator.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary with model names as keys and trained models as values
            Example: {'MLR': mlr_model, 'XGBRF': xgbrf_model, 'LEMBAS-RNN': rnn_model}
        X_data : np.ndarray or pd.DataFrame
            Feature matrix for computing SHAP values
        feature_names : list, optional
            Names of features
        background_samples : int
            Number of background samples for KernelExplainer (default: 100)
        """
        self.models = models_dict
        self.X_data = X_data if isinstance(X_data, pd.DataFrame) else pd.DataFrame(X_data)
        self.feature_names = feature_names if feature_names else [f"Feature_{i}" for i in range(X_data.shape[1])]
        self.X_data.columns = self.feature_names
        self.background_samples = background_samples
        self.shap_values = {}
        self.explainers = {}
        
    def compute_shap_values(self, model_name, model_type='auto'):
        """
        Compute SHAP values for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model (key in models_dict)
        model_type : str
            Type of explainer to use: 'linear', 'tree', 'deep', 'kernel', or 'auto'
            
        Returns:
        --------
        shap_values : np.ndarray
            Computed SHAP values for the model
        """
        model = self.models[model_name]
        
        print(f"Computing SHAP values for {model_name}...")
        
        try:
            if model_type == 'linear' or (model_type == 'auto' and 'MLR' in model_name.upper()):
                # For linear models (MLR)
                explainer = shap.LinearExplainer(model, self.X_data)
                shap_values = explainer.shap_values(self.X_data)
                
            elif model_type == 'tree' or (model_type == 'auto' and 'XGB' in model_name.upper()):
                # For tree-based models (XGBRF)
                # Try TreeExplainer with workaround for XGBoost 2.0+ compatibility issue
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self.X_data)
                except (ValueError, AttributeError) as tree_error:
                    # Known issue with XGBoost 2.0+ and SHAP compatibility
                    if "could not convert string to float" in str(tree_error) or "base_score" in str(tree_error):
                        print(f"  TreeExplainer failed due to XGBoost 2.0+ compatibility issue")
                        print(f"  Trying alternative method: Explainer with model_output='raw'...")
                        
                        try:
                            # Alternative 1: Try with model_output parameter
                            explainer = shap.TreeExplainer(model, model_output='raw')
                            shap_values = explainer.shap_values(self.X_data)
                        except Exception as e2:
                            print(f"  Alternative 1 failed: {e2}")
                            print(f"  Trying KernelExplainer (slower but robust)...")
                            
                            # Alternative 2: Fall back to KernelExplainer
                            background = shap.sample(self.X_data, min(self.background_samples, len(self.X_data)))
                            explainer = shap.KernelExplainer(model.predict, background)
                            shap_values = explainer.shap_values(self.X_data)
                            print(f"  ✓ Using KernelExplainer as fallback")
                    else:
                        raise tree_error
                
            elif model_type == 'deep' or (model_type == 'auto' and 'RNN' in model_name.upper()):
                # For deep learning models (LEMBAS-RNN)
                # Create background dataset for DeepExplainer
                background = self.X_data.sample(n=min(self.background_samples, len(self.X_data)), 
                                               random_state=42)
                
                # Try DeepExplainer first (faster for neural networks)
                try:
                    explainer = shap.DeepExplainer(model, background.values)
                    shap_values = explainer.shap_values(self.X_data.values)
                except Exception as e:
                    print(f"  DeepExplainer failed, falling back to KernelExplainer: {e}")
                    # Fallback to KernelExplainer (model-agnostic but slower)
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(self.X_data)
                    
            elif model_type == 'kernel':
                # Model-agnostic explainer (slowest but most flexible)
                background = shap.sample(self.X_data, self.background_samples)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(self.X_data)
                
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            self.explainers[model_name] = explainer
            self.shap_values[model_name] = shap_values
            print(f"✓ SHAP values computed for {model_name}")
            
            return shap_values  # ← FIXED: Now returns the computed SHAP values
            
        except Exception as e:
            print(f"✗ Error computing SHAP values for {model_name}: {e}")
            raise
    
    def compute_all_shap_values(self):
        """Compute SHAP values for all models."""
        for model_name in self.models.keys():
            self.compute_shap_values(model_name, model_type='auto')
    
    def plot_comparative_summary(self, max_display=20, figsize=(15, 8), save_path=None):
        """
        Create side-by-side summary plots for all models.
        
        Parameters:
        -----------
        max_display : int
            Maximum number of features to display
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        n_models = len(self.shap_values)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, shap_vals) in enumerate(self.shap_values.items()):
            plt.sca(axes[idx])
            shap.summary_plot(shap_vals, self.X_data, 
                            max_display=max_display,
                            show=False,
                            plot_type="dot")
            axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_comparative_importance(self, top_n=15, figsize=(12, 8), save_path=None):
        """
        Create a bar plot comparing mean absolute SHAP values across models.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        importance_data = []
        
        for model_name, shap_vals in self.shap_values.items():
            # Handle multi-output case (take first output if exists)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            
            for feat_idx, feat_name in enumerate(self.feature_names):
                importance_data.append({
                    'Model': model_name,
                    'Feature': feat_name,
                    'Mean |SHAP|': mean_abs_shap[feat_idx]
                })
        
        df_importance = pd.DataFrame(importance_data)
        
        # Get top features based on maximum importance across all models
        top_features = (df_importance.groupby('Feature')['Mean |SHAP|']
                       .max()
                       .sort_values(ascending=False)
                       .head(top_n)
                       .index.tolist())
        
        df_plot = df_importance[df_importance['Feature'].isin(top_features)]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Pivot for grouped bar chart
        df_pivot = df_plot.pivot(index='Feature', columns='Model', values='Mean |SHAP|')
        df_pivot = df_pivot.reindex(top_features)
        
        df_pivot.plot(kind='barh', ax=ax, width=0.8)
        
        ax.set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Comparison Across Models', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(title='Model', title_fontsize=11, fontsize=10, loc='lower right')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return df_pivot
    
    def plot_feature_comparison_heatmap(self, figsize=(10, 8), save_path=None):
        """
        Create a heatmap comparing feature importance across models.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        importance_matrix = []
        model_names = []
        
        for model_name, shap_vals in self.shap_values.items():
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            importance_matrix.append(mean_abs_shap)
            model_names.append(model_name)
        
        df_heatmap = pd.DataFrame(importance_matrix, 
                                 index=model_names, 
                                 columns=self.feature_names)
        
        # Normalize by row to show relative importance within each model
        df_normalized = df_heatmap.div(df_heatmap.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(df_normalized, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': '% of Total Importance'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title('Relative Feature Importance (%) Across Models', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Models', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return df_normalized
    
    def plot_dependence_comparison(self, feature_name, figsize=(15, 4), save_path=None):
        """
        Create dependence plots for a specific feature across all models.
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        n_models = len(self.shap_values)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        feature_idx = self.feature_names.index(feature_name)
        
        for idx, (model_name, shap_vals) in enumerate(self.shap_values.items()):
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            plt.sca(axes[idx])
            shap.dependence_plot(feature_idx, shap_vals, self.X_data,
                               show=False)
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'Dependence Plot: {feature_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def generate_comparison_report(self, output_path='shap_comparison_report.txt'):
        """
        Generate a text report comparing feature importance across models.
        """
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SHAP Feature Contribution Analysis - Model Comparison Report\n")
            f.write("="*80 + "\n\n")
            
            for model_name, shap_vals in self.shap_values.items():
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Mean |SHAP|': mean_abs_shap
                }).sort_values('Mean |SHAP|', ascending=False)
                
                f.write(f"\n{model_name}\n")
                f.write("-"*80 + "\n")
                f.write(feature_importance.to_string(index=False))
                f.write("\n\n")
                
                # Calculate percentage contribution
                total_importance = mean_abs_shap.sum()
                f.write(f"Top 5 features account for: "
                       f"{feature_importance.head(5)['Mean |SHAP|'].sum()/total_importance*100:.2f}% "
                       f"of total importance\n")
        
        print(f"Report saved to {output_path}")
