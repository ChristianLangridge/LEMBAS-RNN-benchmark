"""
Efficient Storage and Retrieval of SHAP Values
==============================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


class SHAPValueStorage:
    """
    Efficient storage manager for SHAP values with multiple backend options.
    
    Supports:
    - Single instance retrieval
    - Batch loading
    - Memory-mapped access for large datasets
    - Metadata storage
    """
    
    def __init__(self, base_path='/home/christianl/Zhang-Lab/Zhang Lab Data/Saved SHAP values'):
        """
        Initialize storage manager.
        
        Parameters:
        -----------
        base_path : str
            Base directory for all SHAP value storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        
    def save_as_memmap(self, shap_values, model_name, metadata=None):
        """
        Save SHAP values as memory-mapped NumPy array.
        
        **BEST FOR:**
        - Single instance retrieval (instant access)
        - Large datasets that don't fit in RAM
        - Random access patterns
        
        **PROS:**
        - Instant loading (no deserialization)
        - Direct single-instance access: shap_array[instance_idx]
        - Minimal memory footprint
        
        **CONS:**
        - Slightly larger file size than compressed formats
        - Less portable across systems
        
        Parameters:
        -----------
        shap_values : np.ndarray
            Shape: (n_samples, n_features) or (n_samples, n_features, n_outputs)
        model_name : str
            Name of model (e.g., 'MLR', 'XGBRF')
        metadata : dict, optional
            Additional information to store
        """
        output_dir = self.base_path / model_name
        output_dir.mkdir(exist_ok=True)
        
        # Save SHAP values as memory-mapped array
        memmap_path = output_dir / 'shap_values.npy'
        np.save(str(memmap_path), shap_values)
        
        # Save metadata separately
        meta = {
            'shape': shap_values.shape,
            'dtype': str(shap_values.dtype),
            'model_name': model_name,
            'n_samples': shap_values.shape[0],
            'n_features': shap_values.shape[1],
        }
        if metadata:
            meta.update(metadata)
        
        meta_path = output_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✓ Saved {model_name} SHAP values as memory-mapped array")
        print(f"  Location: {memmap_path}")
        print(f"  Shape: {shap_values.shape}")
        print(f"  Size: {memmap_path.stat().st_size / (1024**2):.2f} MB")
        
        return memmap_path
    
    def load_memmap(self, model_name, mmap_mode='r'):
        """
        Load SHAP values as memory-mapped array.
        
        Parameters:
        -----------
        model_name : str
            Name of model
        mmap_mode : str
            'r' = read-only (recommended)
            'r+' = read-write
            'c' = copy-on-write
        
        Returns:
        --------
        shap_values : np.memmap
            Memory-mapped array that behaves like numpy array
        metadata : dict
            Associated metadata
        """
        memmap_path = self.base_path / model_name / 'shap_values.npy'
        meta_path = self.base_path / model_name / 'metadata.json'
        
        # Load as memory-mapped array (instant, no RAM usage)
        shap_values = np.load(str(memmap_path), mmap_mode=mmap_mode)
        
        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Loaded {model_name} SHAP values (memory-mapped)")
        print(f"  Shape: {shap_values.shape}")
        print(f"  Mode: {mmap_mode}")
        
        return shap_values, metadata
    
    def get_single_instance_memmap(self, model_name, instance_idx):
        """
        Retrieve SHAP values for a single instance (instant access).
        
        Parameters:
        -----------
        model_name : str
            Name of model
        instance_idx : int
            Index of instance to retrieve
        
        Returns:
        --------
        instance_shap : np.ndarray
            SHAP values for single instance
        """
        shap_values, _ = self.load_memmap(model_name)
        return shap_values[instance_idx]
    