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
        - Requires exact knowledge of shape/dtype for loading
        - Not human-readable
        """
        model_dir = self.base_path / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save memmap
        memmap_path = model_dir / 'shap_values.npy'
        np.save(memmap_path, shap_values)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'shape': shap_values.shape,
            'dtype': str(shap_values.dtype),
            'storage_type': 'memmap'
        })
        
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved {model_name} SHAP values to {memmap_path}")
        print(f"  Shape: {shap_values.shape}, Dtype: {shap_values.dtype}")
        
    
    def load_as_memmap(self, model_name):
        """
        Load SHAP values as memory-mapped array.
        
        Returns:
        --------
        shap_array : np.memmap
            Memory-mapped SHAP values
        metadata : dict
            Associated metadata
        """
        model_dir = self.base_path / model_name
        memmap_path = model_dir / 'shap_values.npy'
        metadata_path = model_dir / 'metadata.json'
        
        # Load metadata to get shape/dtype
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        shape = tuple(metadata['shape'])
        dtype = np.dtype(metadata['dtype'])
        
        # Load as memmap
        shap_array = np.load(memmap_path, mmap_mode='r')
        
        return shap_array, metadata
