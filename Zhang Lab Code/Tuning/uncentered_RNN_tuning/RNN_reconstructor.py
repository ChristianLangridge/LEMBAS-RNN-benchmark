"""
STANDALONE LOADER - EXACT PARAMETERS FROM TRAINING
===================================================

This uses the EXACT initialization parameters from benchmark.py
to ensure perfect reconstruction of your trained model.

Key parameters extracted from benchmark.py:
- projection_amplitude_in = 1.2 (not 1.0!)
- projection_amplitude_out = 1.2 (not 1.0!)
- bionet_params: target_steps=150, max_steps=10, exp_factor=50, tolerance=1e-20, leak=1e-2
- Network columns: source='TF', target='Gene', weight='Interaction'
- Data: ligand_input from 'TF.tsv', tf_output from 'Geneexpression.tsv'
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
# ACTIVATION FUNCTIONS (from activation_functions.py)
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
# UTILITIES (from utilities.py and model_utilities.py)
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
    Format network with Interaction column.
    From model_utilities.py in LEMBAS.
    """
    formatted_net = net.copy()
    
    # Ensure Interaction column exists and is numeric
    if weight_label not in formatted_net.columns:
        raise ValueError(f"Input data must contain `{weight_label}` column")
    formatted_net[weight_label] = pd.to_numeric(formatted_net[weight_label], errors='coerce')
    
    return formatted_net


# =============================================================================
# MODEL CLASSES (from bionetwork.py)
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
        """Scale weights according to spectral radius - used in benchmark.py"""
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
    """Complete signaling network model"""
    
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
        
        # Handle X_in - works with DataFrames, numpy arrays, or Tensors
        if isinstance(X_in, torch.Tensor):
            self.X_in = X_in
        elif isinstance(X_in, pd.DataFrame):
            self.X_in = torch.tensor(X_in.values.copy(), dtype=dtype, device=device)
        else:  # numpy array
            self.X_in = torch.tensor(X_in.copy() if hasattr(X_in, 'copy') else X_in, dtype=dtype, device=device)
        
        # Handle y_out - works with DataFrames, numpy arrays, or Tensors
        if isinstance(y_out, torch.Tensor):
            self.y_out = y_out
        elif isinstance(y_out, pd.DataFrame):
            self.y_out = torch.tensor(y_out.values.copy(), dtype=dtype, device=device)
        else:  # numpy array
            self.y_out = torch.tensor(y_out.copy() if hasattr(y_out, 'copy') else y_out, dtype=dtype, device=device)
        
        # Handle input labels
        if isinstance(X_in, pd.DataFrame):
            input_labels = X_in.columns.values
        else:  # numpy array or Tensor
            input_labels = np.arange(self.X_in.shape[1])
        
        # Handle output labels
        if isinstance(y_out, pd.DataFrame):
            output_labels = y_out.columns.values
        else:  # numpy array or Tensor
            output_labels = np.arange(self.y_out.shape[1])
        
        self.input_layer = ProjectInput(node_idx_map=self.node_idx_map,
                                       input_labels=input_labels,
                                       projection_amplitude=projection_amplitude_in,
                                       dtype=self.dtype, device=self.device)
        
        self.signaling_network = BioNet(edge_list=edge_list, edge_MOA=edge_MOA,
                                       n_network_nodes=len(node_labels),
                                       bionet_params=bionet_params,
                                       activation_function=activation_function,
                                       dtype=self.dtype, device=self.device, seed=self.seed)
        
        self.output_layer = ProjectOutput(node_idx_map=self.node_idx_map,
                                         output_labels=output_labels,
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
    
    def forward(self, X_in):
        X_full = self.input_layer(X_in)
        Y_full = self.signaling_network(X_full)
        Y_hat = self.output_layer(Y_full)
        return Y_hat, Y_full
    
    def copy(self):
        return copy.deepcopy(self)


# =============================================================================
# UTILITY FUNCTION FOR COLUMN DETECTION
# =============================================================================

def detect_network_columns(net: pd.DataFrame):
    """
    Automatically detect column names in the network DataFrame.
    
    Returns
    -------
    source_col : str
        Name of the source column
    target_col : str
        Name of the target column
    weight_col : str
        Name of the weight/interaction column
    """
    
    cols = net.columns.tolist()
    print(f"\nDetected network columns: {cols}")
    
    # Common patterns for source column
    source_patterns = ['source', 'tf', 'from', 'src', 'regulator']
    source_col = None
    for pattern in source_patterns:
        matches = [c for c in cols if pattern.lower() in c.lower()]
        if matches:
            source_col = matches[0]
            break
    
    # Common patterns for target column
    target_patterns = ['target', 'gene', 'to', 'tgt', 'dest', 'destination']
    target_col = None
    for pattern in target_patterns:
        matches = [c for c in cols if pattern.lower() in c.lower()]
        if matches:
            target_col = matches[0]
            break
    
    # Common patterns for weight column
    weight_patterns = ['interaction', 'weight', 'mode', 'moa', 'sign', 'type', 'effect']
    weight_col = None
    for pattern in weight_patterns:
        matches = [c for c in cols if pattern.lower() in c.lower()]
        if matches:
            weight_col = matches[0]
            break
    
    # If still not found, use positional defaults
    if source_col is None and len(cols) >= 1:
        source_col = cols[0]
        print(f"⚠️  Using first column as source: '{source_col}'")
    
    if target_col is None and len(cols) >= 2:
        target_col = cols[1]
        print(f"⚠️  Using second column as target: '{target_col}'")
    
    if weight_col is None and len(cols) >= 3:
        weight_col = cols[2]
        print(f"⚠️  Using third column as weight: '{weight_col}'")
    
    if source_col is None or target_col is None or weight_col is None:
        raise ValueError(f"Could not detect network columns. Found: {cols}")
    
    print(f"✓ Using columns: source='{source_col}', target='{target_col}', weight='{weight_col}'")
    
    return source_col, target_col, weight_col


# =============================================================================
# CHECKPOINT LOADER WITH EXACT TRAINING PARAMETERS
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path, 
    node_names=None, 
    net=None, 
    X_in=None, 
    y_out=None, 
    device='cpu',
    use_exact_training_params=True
):
    """
    Load model from checkpoint using EXACT parameters from benchmark.py
    
    Parameters
    ----------
    checkpoint_path : str
        Path to your .pt checkpoint file
    node_names : list, optional
        Node names. If None, uses generic names.
    net : pd.DataFrame, optional
        Network with columns: 'TF' (source), 'Gene' (target), 'Interaction' (weight)
        If None, reconstructs from checkpoint.
    X_in : pd.DataFrame, optional
        Input data (TF activities). If None, creates dummy data.
    y_out : pd.DataFrame, optional
        Output data (gene expression). If None, creates dummy data.
    device : str
        'cpu' or 'cuda'
    use_exact_training_params : bool
        If True, uses EXACT params from benchmark.py. If False, uses checkpoint params.
        
    Returns
    -------
    model : SignalingModel
        Loaded model ready to use
        
    Notes
    -----
    EXACT TRAINING PARAMETERS from benchmark.py:
    - projection_amplitude_in = 1.2
    - projection_amplitude_out = 1.2
    - bionet_params = {
        'target_steps': 150,
        'max_steps': 10,
        'exp_factor': 50,
        'tolerance': 1e-20,
        'leak': 1e-2
      }
    - weight_label = 'Interaction'
    - source_label = 'TF'
    - target_label = 'Gene'
    - prescale_weights called with target_radius=0.8
    - input_layer.weights.requires_grad = False
    """
    
    print("=" * 70)
    print("LOADING WITH EXACT TRAINING PARAMETERS")
    print("=" * 70)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # EXACT parameters from benchmark.py
    if use_exact_training_params:
        print("\n✓ Using EXACT parameters from benchmark.py")
        config = {
            'seed': checkpoint.get('seed', 888),
            'projection_amplitude_in': 1.2,  # From benchmark.py!
            'projection_amplitude_out': 1.2,  # From benchmark.py!
            'bionet_params': {
                'target_steps': 150,  # From benchmark.py!
                'max_steps': 10,      # From benchmark.py!
                'exp_factor': 50,     # From benchmark.py!
                'tolerance': 1e-20,   # From benchmark.py!
                'leak': 1e-2          # From benchmark.py! (same as 0.01)
            }
        }
    else:
        # Use checkpoint params
        config = {
            'seed': checkpoint.get('seed', 888),
            'bionet_params': checkpoint.get('bionet_params'),
            'projection_amplitude_in': checkpoint.get('projection_amplitude_in', 1.0),
            'projection_amplitude_out': checkpoint.get('projection_amplitude_out', 1.0)
        }
    
    print(f"\nConfiguration:")
    print(f"  seed: {config['seed']}")
    print(f"  projection_amplitude_in: {config['projection_amplitude_in']}")
    print(f"  projection_amplitude_out: {config['projection_amplitude_out']}")
    print(f"  bionet_params: {config['bionet_params']}")
    
    # Get dimensions
    network_key = [k for k in state_dict.keys() if 'network' in k.lower() and 'weight' in k.lower() and len(state_dict[k].shape) == 2][0]
    adj_matrix = state_dict[network_key].cpu().numpy()
    n_nodes = adj_matrix.shape[0]
    
    input_key = [k for k in state_dict.keys() if 'input' in k.lower() and 'weight' in k.lower()][0]
    n_ligands = state_dict[input_key].shape[0]
    
    output_key = [k for k in state_dict.keys() if 'output' in k.lower() and 'weight' in k.lower()][0]
    n_tfs = state_dict[output_key].shape[0]
    
    print(f"\nModel dimensions:")
    print(f"  Total nodes: {n_nodes}")
    print(f"  Input TFs: {n_ligands}")
    print(f"  Output genes: {n_tfs}")
    print(f"  Edges: {np.count_nonzero(adj_matrix)}")
    
    # Create node names
    if node_names is None:
        node_names = [f'node_{i}' for i in range(n_nodes)]
        print("\n⚠️  Using generic node names (node_0, node_1, ...)")
    
    # Reconstruct network with EXACT column names from benchmark.py
    if net is None:
        sources, targets, weights = [], [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] != 0:
                    targets.append(node_names[i])
                    sources.append(node_names[j])
                    weights.append(adj_matrix[i, j])
        
        # Use benchmark.py column names: 'TF', 'Gene', 'Interaction'
        net = pd.DataFrame({
            'TF': sources,        # source_label from benchmark.py
            'Gene': targets,      # target_label from benchmark.py
            'Interaction': [1 if w > 0 else -1 for w in weights]  # weight_label from benchmark.py
        })
        print(f"\n✓ Reconstructed network: {len(net)} edges")
        print(f"  Using benchmark.py columns: 'TF', 'Gene', 'Interaction'")
    
    # Format network (from benchmark.py)
    net = format_network(net, weight_label='Interaction')
    
    # Create dummy data
    if X_in is None:
        X_in = pd.DataFrame(
            np.random.rand(100, n_ligands) * 0.1,
            columns=[node_names[i] for i in range(n_ligands)]
        )
        print("\n⚠️  Using dummy input data")
    
    if y_out is None:
        y_out = pd.DataFrame(
            np.random.rand(100, n_tfs) * 0.1,
            columns=[node_names[i] for i in range(n_tfs)]
        )
        print("⚠️  Using dummy output data")
    
    # Initialize model with EXACT parameters from benchmark.py
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    model = SignalingModel(
        net=net,
        X_in=X_in,
        y_out=y_out,
        projection_amplitude_in=config['projection_amplitude_in'],
        projection_amplitude_out=config['projection_amplitude_out'],
        weight_label='Interaction',  # From benchmark.py
        source_label='TF',           # From benchmark.py
        target_label='Gene',         # From benchmark.py
        bionet_params=config['bionet_params'],
        activation_function='MML',
        dtype=torch.float32,
        device=device,
        seed=config['seed']
    )
    
    print("✓ Model structure initialized")
    
    # Load weights
    print("\n" + "=" * 70)
    print("LOADING WEIGHTS")
    print("=" * 70)
    
    model.load_state_dict(state_dict)
    print("✓ Weights loaded")
    
    # Apply settings from benchmark.py
    print("\n" + "=" * 70)
    print("APPLYING BENCHMARK.PY SETTINGS")
    print("=" * 70)
    
    # From benchmark.py: input_layer.weights.requires_grad = False
    model.input_layer.weights.requires_grad = False
    print("✓ Set input_layer.weights.requires_grad = False")
    
    # Note: prescale_weights was called BEFORE training, so weights are already scaled
    # The loaded weights from checkpoint already have this applied
    print("✓ Weights already include prescale_weights(target_radius=0.8) from training")
    
    print("\n" + "=" * 70)
    print("✅ MODEL LOADED WITH EXACT TRAINING PARAMETERS")
    print("=" * 70)
    
    return model
