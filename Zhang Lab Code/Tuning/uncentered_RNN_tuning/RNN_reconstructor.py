"""
 RNN RECONSTRUCTOR - REINITIALISES RNN FROM CHECKPOINT 
==============================================================

This version replicates the behavior from benchmark.py training script:
1. Passes df (not tensors) to SignalingModel
2. Allows SignalingModel to filter data to network nodes internally
3. Converts to tensors after model initialisation
4. Uses format_network() exactly as in training


"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigs
from typing import Dict, Union, List
import copy


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def MML_activation(x: torch.Tensor, leak: Union[float, int]):
    """Michaelis-Menten-like activation function"""
    fx = torch.nn.functional.leaky_relu(input=x, negative_slope=leak, inplace=False)
    shifted_x = 0.5 * (fx - 0.5)
    mask = torch.lt(shifted_x, 0.0)
    gated_x = fx + 10 * mask
    right_values = 0.5 + torch.div(shifted_x, gated_x)
    fx = mask * (fx - right_values) + right_values
    return fx


def MML_delta_activation(x: torch.Tensor, leak: Union[float, int]):
    """Derivative of MML activation"""
    mask1 = x.le(0)
    y = torch.ones(x.shape, dtype=x.dtype, device=x.device)
    mask2 = x.gt(0.5)
    right_values = 0.25/torch.pow(x + 1e-12, 2) - 1
    y = y + mask2 * right_values
    y = y - (1-leak) * mask1
    return y


def MML_onestepdelta_activation_factor(Y_full: torch.Tensor, leak: Union[float, int]=0.01):
    """Adjusts weights for linearization in spectral radius"""
    y = torch.ones_like(Y_full)
    piece1 = Y_full.le(0)
    piece3 = Y_full.gt(0.5)
    safe_x = torch.clamp(1-Y_full, max=0.9999)
    right_values = 4 * torch.pow(safe_x, 2) - 1
    y = y + piece3 * right_values
    y = y - (1-leak) * piece1
    return y


activation_function_map = {
    'MML': {
        'activation': MML_activation,
        'delta': MML_delta_activation,
        'onestepdelta': MML_onestepdelta_activation_factor
    }
}


# =============================================================================
# UTILITIES
# =============================================================================

def set_seeds(seed: int=888):
    """Set random seeds for reproducibility"""
    import os
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ.keys():
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def np_to_torch(arr: np.array, dtype: torch.dtype, device: str = 'cpu'):
    """Convert numpy array to torch tensor"""
    return torch.tensor(arr, dtype=dtype, device=device)


def format_network(net: pd.DataFrame, weight_label: str = 'Interaction') -> pd.DataFrame:
    """
    Format network - EXACTLY as in training script.
    Only ensures weight column is numeric.
    """
    formatted_net = net.copy()
    
    if weight_label not in formatted_net.columns:
        raise ValueError(f"Input data must contain `{weight_label}` column")
    formatted_net[weight_label] = pd.to_numeric(formatted_net[weight_label], errors='coerce')
    
    return formatted_net


# =============================================================================
# MODEL CLASSES
# =============================================================================

class ProjectInput(nn.Module):
    """Project input ligands to full network"""
    
    def __init__(self, node_idx_map: Dict[str, int], input_labels: np.array,
                 projection_amplitude: Union[int, float] = 1,
                 dtype: torch.dtype=torch.float32, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.projection_amplitude = projection_amplitude
        self.size_out = len(node_idx_map)
        self.input_node_order = torch.tensor([node_idx_map[x] for x in input_labels], device=self.device)
        weights = self.projection_amplitude * torch.ones(len(input_labels), dtype=self.dtype, device=self.device)
        self.weights = nn.Parameter(weights)
    
    def forward(self, X_in: torch.Tensor):
        X_full = torch.zeros([X_in.shape[0], self.size_out], dtype=self.dtype, device=self.device)
        X_full[:, self.input_node_order] = self.weights * X_in
        return X_full


class BioNet(nn.Module):
    """RNN on signaling network topology"""
    
    def __init__(self, edge_list: np.array, edge_MOA: np.array, n_network_nodes: int,
                 bionet_params: Dict[str, float], activation_function: str = 'MML',
                 dtype: torch.dtype=torch.float32, device: str = 'cpu', seed: int = 888):
        super().__init__()
        self.training_params = bionet_params
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._ss_seed_counter = 0
        self.n_network_nodes = n_network_nodes
        
        self.edge_list = (np_to_torch(edge_list[0,:], dtype=torch.int32, device='cpu'),
                         np_to_torch(edge_list[1,:], dtype=torch.int32, device='cpu'))
        self.edge_MOA = np_to_torch(edge_MOA, dtype=torch.bool, device=self.device)
        
        weights, bias = self.initialize_weights()
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        
        self.weights_MOA, self.mask_MOA = self.make_mask_MOA()
        self.activation = activation_function_map[activation_function]['activation']
        self.delta = activation_function_map[activation_function]['delta']
        self.onestepdelta_activation_factor = activation_function_map[activation_function]['onestepdelta']
    
    def initialize_weight_values(self):
        network_targets = self.edge_list[0].numpy()
        n_interactions = len(network_targets)
        set_seeds(self.seed)
        weight_values = 0.1 + 0.1*torch.rand(n_interactions, dtype=self.dtype, device=self.device)
        weight_values[self.edge_MOA[1,:]] = -weight_values[self.edge_MOA[1,:]]
        bias = 1e-3*torch.ones((self.n_network_nodes, 1), dtype=self.dtype, device=self.device)
        for nt_idx in np.unique(network_targets):
            if torch.all(weight_values[network_targets == nt_idx]<0):
                bias.data[nt_idx] = 1
        return weight_values, bias
    
    def make_mask(self):
        weights_mask = torch.zeros(self.n_network_nodes, self.n_network_nodes, dtype=bool, device=self.device)
        weights_mask[self.edge_list] = True
        weights_mask = torch.logical_not(weights_mask)
        return weights_mask
    
    def initialize_weights(self):
        weight_values, bias = self.initialize_weight_values()
        self.mask = self.make_mask()
        weights = torch.zeros(self.mask.shape, dtype=self.dtype, device=self.device)
        weights[self.edge_list] = weight_values
        return weights, bias
    
    def make_mask_MOA(self):
        signed_MOA = self.edge_MOA[0, :].type(torch.long) - self.edge_MOA[1, :].type(torch.long)
        weights_MOA = torch.zeros(self.n_network_nodes, self.n_network_nodes, dtype=torch.long, device=self.device)
        weights_MOA[self.edge_list] = signed_MOA
        mask_MOA = weights_MOA == 0
        return weights_MOA, mask_MOA
    
    def forward(self, X_full: torch.Tensor):
        X_bias = X_full.T + self.bias
        X_new = torch.zeros_like(X_bias)
        
        for t in range(self.training_params['max_steps']):
            X_old = X_new
            X_new = torch.mm(self.weights, X_new)
            X_new = X_new + X_bias
            X_new = self.activation(X_new, self.training_params['leak'])
            
            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))
                if diff.lt(self.training_params['tolerance']):
                    break
        
        Y_full = X_new.T
        return Y_full
    
    def prescale_weights(self, target_radius: float = 0.8):
        """Scale weights according to spectral radius"""
        A = scipy.sparse.csr_matrix(self.weights.detach().cpu().numpy())
        np.random.seed(self.seed)
        eigen_value, _ = eigs(A, k=1, v0=np.random.rand(A.shape[0]))
        spectral_radius = np.abs(eigen_value)
        
        factor = target_radius/spectral_radius.item()
        self.weights.data = self.weights.data * factor


class ProjectOutput(nn.Module):
    """Project network to TF outputs"""
    
    def __init__(self, node_idx_map: Dict[str, int], output_labels: np.array,
                 projection_amplitude: Union[int, float] = 1,
                 dtype: torch.dtype=torch.float32, device: str = 'cpu'):
        super().__init__()
        self.size_in = len(node_idx_map)
        self.size_out = len(output_labels)
        self.projection_amplitude = projection_amplitude
        self.output_node_order = torch.tensor([node_idx_map[x] for x in output_labels], device=device)
        weights = self.projection_amplitude * torch.ones(len(output_labels), dtype=dtype, device=device)
        self.weights = nn.Parameter(weights)
        bias = torch.zeros(len(output_labels)).to(device, dtype)
        self.bias = nn.Parameter(bias)
    
    def forward(self, Y_full):
        Y_hat = (self.weights * Y_full[:, self.output_node_order]) + self.bias
        return Y_hat


class SignalingModel(torch.nn.Module):
    """Complete signaling network model - MATCHES TRAINING SCRIPT"""
    
    DEFAULT_TRAINING_PARAMETERS = {
        'target_steps': 100,
        'max_steps': 300,
        'exp_factor': 20,
        'leak': 0.01,
        'tolerance': 1e-5
    }
    
    def __init__(self, net: pd.DataFrame, X_in: pd.DataFrame, y_out: pd.DataFrame,
                 projection_amplitude_in: Union[int, float] = 1,
                 projection_amplitude_out: float = 1,
                 ban_list: List[str] = None,
                 weight_label: str = 'mode_of_action',
                 source_label: str = 'source',
                 target_label: str = 'target',
                 bionet_params: Dict[str, float] = None,
                 activation_function: str='MML',
                 dtype: torch.dtype=torch.float32,
                 device: str = 'cpu',
                 seed: int = 888):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._gradient_seed_counter = 0
        self.projection_amplitude_out = projection_amplitude_out
        
        edge_list, node_labels, edge_MOA = self.parse_network(net, ban_list, weight_label, source_label, target_label)
        
        if not bionet_params:
            bionet_params = self.DEFAULT_TRAINING_PARAMETERS.copy()
        else:
            bionet_params = self.set_training_parameters(**bionet_params)
        
        # CRITICAL: filters data to network nodes like training run benchmark.py
        self.X_in = X_in.loc[:, np.intersect1d(X_in.columns.values, node_labels)]
        self.y_out = y_out.loc[:, np.intersect1d(y_out.columns.values, node_labels)]
        
        print(f"  Filtered X_in: {X_in.shape[1]} → {self.X_in.shape[1]} features")
        print(f"  Filtered y_out: {y_out.shape[1]} → {self.y_out.shape[1]} features")
        
        self.input_layer = ProjectInput(node_idx_map=self.node_idx_map,
                                       input_labels=self.X_in.columns.values,
                                       projection_amplitude=projection_amplitude_in,
                                       dtype=self.dtype, device=self.device)
        
        self.signaling_network = BioNet(edge_list=edge_list, edge_MOA=edge_MOA,
                                       n_network_nodes=len(node_labels),
                                       bionet_params=bionet_params,
                                       activation_function=activation_function,
                                       dtype=self.dtype, device=self.device, seed=self.seed)
        
        self.output_layer = ProjectOutput(node_idx_map=self.node_idx_map,
                                         output_labels=self.y_out.columns.values,
                                         projection_amplitude=self.projection_amplitude_out,
                                         dtype=self.dtype, device=device)
    
    def parse_network(self, net: pd.DataFrame, ban_list: List[str] = None,
                     weight_label: str = 'mode_of_action',
                     source_label: str = 'source',
                     target_label: str = 'target'):
        if not ban_list:
            ban_list = []
        
        net = net[~net[source_label].isin(ban_list)]
        net = net[~net[target_label].isin(ban_list)]
        
        node_labels = sorted(pd.concat([net[source_label], net[target_label]]).unique())
        self.node_idx_map = {node_name: idx for idx, node_name in enumerate(node_labels)}
        
        source_indices = net[source_label].map(self.node_idx_map).values
        target_indices = net[target_label].map(self.node_idx_map).values
        
        n_nodes = len(node_labels)
        A = scipy.sparse.csr_matrix((net[weight_label].values, (source_indices, target_indices)), shape=(n_nodes, n_nodes))
        source_indices, target_indices, edge_MOA = scipy.sparse.find(A)
        edge_list = np.array((target_indices, source_indices))
        edge_MOA = np.array([[edge_MOA==1], [edge_MOA==-1]]).squeeze()
        
        return edge_list, node_labels, edge_MOA
    
    def set_training_parameters(self, **attributes):
        default_parameters = self.DEFAULT_TRAINING_PARAMETERS.copy()
        allowed_params = list(default_parameters.keys()) + ['spectral_target']
        params = {**default_parameters, **attributes}
        if 'spectral_target' not in params.keys():
            params['spectral_target'] = np.exp(np.log(params['tolerance'])/params['target_steps'])
        params = {k: v for k, v in params.items() if k in allowed_params}
        return params
    
    def df_to_tensor(self, df: pd.DataFrame):
        """Convert DataFrame to tensor - EXACTLY as in training script"""
        return torch.tensor(df.values.copy(), dtype=self.dtype, device=self.device)
    
    def forward(self, X_in):
        X_full = self.input_layer(X_in)
        Y_full = self.signaling_network(X_full)
        Y_hat = self.output_layer(Y_full)
        return Y_hat, Y_full
    
    def copy(self):
        return copy.deepcopy(self)


# =============================================================================
# CHECKPOINT LOADER 
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path,
    net_path,
    X_in_df,
    y_out_df,
    device='cpu',
    use_exact_training_params=True
):
    """
    Load model EXACTLY as training script does.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint
    net_path : str
        Path to network.tsv file
    X_in_df : pd.DataFrame
        Input DataFrame (will be automatically filtered to network nodes)
    y_out_df : pd.DataFrame
        Output DataFrame (will be automatically filtered to network nodes)
    device : str
        'cpu' or 'cuda'
    use_exact_training_params : bool
        Use exact params from benchmark.py
        
    Returns
    -------
    model : SignalingModel
        Loaded model ready for inference
        
    Notes
    -----
    This replicates the EXACT sequence from benchmark.py:
    1. Load network with pd.read_csv
    2. Format network with format_network()
    3. Pass DataFrames to SignalingModel (NOT tensors!)
    4. Let SignalingModel filter data internally
    5. Convert to tensors AFTER initialization
    6. Set input_layer.weights.requires_grad = False
    7. Apply prescale_weights
    8. Load state dict
    """
    
    print("=" * 70)
    print("LOADING MODEL - EXACT TRAINING SCRIPT SEQUENCE")
    print("=" * 70)
    
    # Step 1: Load checkpoint
    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # Step 2: Load network - EXACTLY as training script
    print(f"\n2. Loading network from: {net_path}")
    net = pd.read_csv(net_path, sep='\t', index_col=False)
    print(f"   Network shape: {net.shape}")
    print(f"   Network columns: {net.columns.tolist()}")
    
    # Step 3: Format network - EXACTLY as training script
    print(f"\n3. Formatting network...")
    net = format_network(net, weight_label='Interaction')
    
    # Step 4: Get configuration
    if use_exact_training_params:
        print(f"\n4. Using EXACT benchmark.py parameters")
        config = {
            'seed': checkpoint.get('seed', 888),
            'projection_amplitude_in': 1.2,
            'projection_amplitude_out': 1.2,
            'bionet_params': {
                'target_steps': 150,
                'max_steps': 10,
                'exp_factor': 50,
                'tolerance': 1e-20,
                'leak': 1e-2
            }
        }
    else:
        config = {
            'seed': checkpoint.get('seed', 888),
            'bionet_params': checkpoint.get('bionet_params'),
            'projection_amplitude_in': checkpoint.get('projection_amplitude_in', 1.0),
            'projection_amplitude_out': checkpoint.get('projection_amplitude_out', 1.0)
        }
    
    print(f"   projection_amplitude_in: {config['projection_amplitude_in']}")
    print(f"   projection_amplitude_out: {config['projection_amplitude_out']}")
    print(f"   bionet_params: {config['bionet_params']}")
    
    # Step 5: Initialize model
    print(f"\n5. Initializing model with DataFrames...")
    print(f"   Input X_in shape: {X_in_df.shape}")
    print(f"   Input y_out shape: {y_out_df.shape}")
    
    model = SignalingModel(
        net=net,
        X_in=X_in_df,  
        y_out=y_out_df, 
        projection_amplitude_in=config['projection_amplitude_in'],
        projection_amplitude_out=config['projection_amplitude_out'],
        weight_label='Interaction',
        source_label='TF',
        target_label='Gene',
        bionet_params=config['bionet_params'],
        activation_function='MML',
        dtype=torch.float32,
        device=device,
        seed=config['seed']
    )
    
    print("   ✓ Model initialized (data automatically filtered)")
    
    # Step 6: Convert to tensors 
    print(f"\n6. Converting DataFrames to tensors...")
    model.X_in = model.df_to_tensor(model.X_in)
    model.y_out = model.df_to_tensor(model.y_out)
    print("   ✓ Tensors created")
    
    # Step 7: Set training flags 
    print(f"\n7. Applying training settings...")
    model.input_layer.weights.requires_grad = False
    print("   ✓ Set input_layer.weights.requires_grad = False")
    
    model.signaling_network.prescale_weights(target_radius=0.8)
    print("   ✓ Applied prescale_weights(target_radius=0.8)")
    
    # Step 8: Load weights
    print(f"\n8. Loading trained weights...")
    model.load_state_dict(state_dict)
    print("   ✓ Weights loaded")
    
    # Step 9: Set to eval mode
    model.eval()
    print("   ✓ Model set to eval mode")
    
    print("\n" + "=" * 70)
    print("✅ MODEL LOADED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nModel ready for inference:")
    print(f"  Input features: {model.X_in.shape[1]}")
    print(f"  Output features: {model.y_out.shape[1]}")
    print(f"  Total network nodes: {model.signaling_network.n_network_nodes}")
    
    return model