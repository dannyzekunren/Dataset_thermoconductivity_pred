"""
MLIP Embedding CNN Regression for Thermal Conductivity Prediction

This script extracts embeddings from orb-v3-conservative-20-omat model and trains
1D CNN architectures to predict log thermal conductivity across 3 data splits.
"""

import pickle
import json
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from orb_models.forcefield import pretrained, atomic_system

# PyTorch Geometric imports for graph neural networks
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
    from torch_geometric.utils import add_self_loops
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. CGCNN model will not be available.")
    print("Install with: pip install torch-geometric")

# TabPFN imports for tabular deep learning
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: TabPFN not available. TabPFN model will not be available.")
    print("Install with: pip install tabpfn")

#TABPFN_AVAILABLE = False
# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


def load_split_data(split_name):
    """Load data from a specific split file."""
    print(f"\nLoading {split_name}...")
    with open(f'processed_splits/{split_name}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_structures = data['X_train']['structures']
    test_structures = data['X_test']['structures']
    y_train = data['y_train_log_klat']
    y_test = data['y_test_log_klat']
    
    # Extract additional properties
    train_props = {
        'formation_energy': data['X_train']['formation_energy_per_atom'],
        'e_above_hull': data['X_train']['e_above_hull'],
        'band_gap': data['X_train']['band_gap'],
        'elasticity_K_VRH': data['X_train']['elasticity_K_VRH'],
        'elasticity_G_VRH': data['X_train']['elasticity_G_VRH']
    }
    
    test_props = {
        'formation_energy': data['X_test']['formation_energy_per_atom'],
        'e_above_hull': data['X_test']['e_above_hull'],
        'band_gap': data['X_test']['band_gap'],
        'elasticity_K_VRH': data['X_test']['elasticity_K_VRH'],
        'elasticity_G_VRH': data['X_test']['elasticity_G_VRH']
    }
    
    print(f"  Train samples: {len(train_structures)}, Test samples: {len(test_structures)}")
    return train_structures, test_structures, y_train, y_test, train_props, test_props


def extract_masked_statistics(embeddings_list, lengths=None):
    """
    Extract masked statistics (mean, std, max) from variable-length node embeddings.
    
    Args:
        embeddings_list: List of numpy arrays of shape [n_atoms, 256]
        lengths: List of actual atom counts (optional, will infer from embeddings if None)
    
    Returns:
        numpy array of shape [n_structures, 768] (256*3 for mean, std, max)
    """
    features = []
    
    for i, node_feats in enumerate(embeddings_list):
        # node_feats: [n_atoms, 256]
        n_atoms = node_feats.shape[0]
        feat_dim = node_feats.shape[1]
        
        # Convert to tensor
        node_tensor = torch.FloatTensor(node_feats)  # [n_atoms, 256]
        
        # Create mask for valid atoms
        if lengths is not None:
            actual_length = lengths[i]
            mask = torch.arange(n_atoms) < actual_length
        else:
            # Use all atoms if lengths not provided
            mask = torch.ones(n_atoms, dtype=torch.bool)
        
        mask_float = mask.float().unsqueeze(-1)  # [n_atoms, 1]
        valid_count = mask_float.sum() + 1e-8
        
        # Masked mean
        mean_feat = (node_tensor * mask_float).sum(0) / valid_count  # [256]
        
        # Masked std
        mean_expanded = mean_feat.unsqueeze(0)  # [1, 256]
        variance = ((node_tensor - mean_expanded) ** 2 * mask_float).sum(0) / valid_count
        std_feat = variance.clamp_min(0).sqrt()  # [256]
        
        # Masked max
        node_masked = node_tensor.clone()
        node_masked[~mask] = -1e9  # Set invalid positions to very negative value
        max_feat = node_masked.max(0).values  # [256]
        
        # Concatenate all statistics
        combined = torch.cat([mean_feat, std_feat, max_feat], dim=0)  # [768]
        features.append(combined.numpy())
    
    return np.array(features)


def extract_embeddings(structures, orbff, device, return_sequences=False, cache_file=None):
    """
    Extract embeddings from structures using the ORB model.
    
    Args:
        structures: List of pymatgen Structure objects
        orbff: Loaded ORB model
        device: torch device
        return_sequences: If True, return per-atom sequences; else return aggregated embeddings
        cache_file: Optional file path to save/load embeddings
    
    Returns:
        If return_sequences=False: numpy array of shape [n_structures, 512] (mean+max pooled)
        If return_sequences=True: list of numpy arrays of shape [n_atoms, 256]
    """
    # Check if cached embeddings exist
    if cache_file and os.path.exists(cache_file):
        print(f"  Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    embeddings = []
    
    print(f"  Extracting {'sequence' if return_sequences else 'aggregated'} embeddings...")
    
    for structure in tqdm(structures, desc="  Processing structures"):
        # Convert pymatgen Structure to ASE Atoms
        atoms = structure.to_ase_atoms()
        
        # Create graph
        graph = atomic_system.ase_atoms_to_atom_graphs(
            atoms, orbff.system_config, device=device
        )
        
        # Extract node features (per-atom embeddings)
        with torch.no_grad():
            node_features = orbff.model(graph)["node_features"]  # [n_atoms, 256]
        
        if return_sequences:
            # Return per-atom embeddings for CNN-B
            embeddings.append(node_features.cpu().numpy())
        else:
            # Aggregate: concatenate mean + max pooling for CNN-A
            mean_emb = node_features.mean(dim=0)  # [256]
            max_emb = node_features.max(dim=0)[0]  # [256]
            agg_emb = torch.cat([mean_emb, max_emb])  # [512]
            embeddings.append(agg_emb.cpu().numpy())
    
    result = embeddings if return_sequences else np.array(embeddings)
    
    # Save to cache
    if cache_file:
        print(f"  Saving embeddings to {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    return result


def build_knn_graph(structure, embeddings, k_neighbors=12):
    """
    Build k-nearest neighbor graph from structure and embeddings.
    
    Args:
        structure: pymatgen Structure object
        embeddings: numpy array of shape [n_atoms, 256] (per-atom embeddings)
        k_neighbors: number of nearest neighbors to consider
    
    Returns:
        edge_index: [2, num_edges] tensor of graph connectivity
        edge_attr: [num_edges, 1] tensor of edge attributes (distances)
    """
    from scipy.spatial import distance_matrix
    
    # Get atomic positions
    positions = structure.cart_coords  # [n_atoms, 3]
    n_atoms = len(positions)
    
    # Compute pairwise distance matrix
    dist_matrix = distance_matrix(positions, positions)
    
    # Build k-nearest neighbor graph
    edge_index = []
    edge_attr = []
    
    for i in range(n_atoms):
        # Get k nearest neighbors (excluding self)
        distances = dist_matrix[i]
        # Get indices of k+1 nearest neighbors (including self)
        nearest_indices = np.argsort(distances)[:k_neighbors + 1]
        
        for j in nearest_indices:
            if i != j:  # Exclude self-loops
                edge_index.append([i, j])
                edge_attr.append([distances[j]])
    
    edge_index = torch.LongTensor(edge_index).t().contiguous()  # [2, num_edges]
    edge_attr = torch.FloatTensor(edge_attr)  # [num_edges, 1]
    
    return edge_index, edge_attr


def extract_graph_data(structures, orbff, device, k_neighbors=12, cache_file=None):
    """
    Extract graph data with embeddings from structures using ORB's native graph.
    
    Args:
        structures: List of pymatgen Structure objects
        orbff: Loaded ORB model
        device: torch device
        k_neighbors: number of nearest neighbors (fallback for k-NN if ORB edges unavailable)
        cache_file: Optional file path to save/load graph data
    
    Returns:
        List of (node_features, edge_index, edge_attr) tuples
    """
    # Check if cached graph data exists
    if cache_file and os.path.exists(cache_file):
        print(f"  Loading cached graph data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    graph_data = []
    
    print(f"  Extracting graph data using ORB's native graph structure...")
    
    for structure in tqdm(structures, desc="  Processing structures"):
        # Convert pymatgen Structure to ASE Atoms
        atoms = structure.to_ase_atoms()
        
        # Create graph for ORB model
        graph = atomic_system.ase_atoms_to_atom_graphs(
            atoms, orbff.system_config, device=device
        )
        
        # Extract node features (per-atom embeddings)
        with torch.no_grad():
            node_features = orbff.model(graph)["node_features"]  # [n_atoms, 256]
        
        node_features_np = node_features.cpu().numpy()
        
        # Try to extract edge information from ORB's native graph
        try:
            # Check for different possible field names
            if "senders" in graph and "receivers" in graph:
                senders = graph["senders"]
                receivers = graph["receivers"]
            elif "edge_src" in graph and "edge_dst" in graph:
                senders = graph["edge_src"]
                receivers = graph["edge_dst"]
            else:
                # Fallback: use k-NN graph
                raise KeyError("No edge indices found in graph")
            
            # Build edge_index [2, num_edges]
            edge_index = torch.stack([senders, receivers], dim=0).cpu()
            
            # Extract edge attributes (distances)
            if "edge_vec" in graph:
                # Use edge vectors to compute distances
                edge_vec = graph["edge_vec"]
                edge_attr = edge_vec.norm(dim=-1, keepdim=True).cpu()  # [num_edges, 1]
            elif "positions" in graph:
                # Compute distances from positions
                positions = graph["positions"]
                edge_attr = (positions[senders] - positions[receivers]).norm(dim=-1, keepdim=True).cpu()
            else:
                # Use unit edge attributes as fallback
                edge_attr = torch.ones(edge_index.size(1), 1)
        
        except (KeyError, AttributeError) as e:
            # Fallback to k-NN graph construction
            print(f"\n  Warning: Could not extract ORB edges ({e}), falling back to k-NN graph")
            edge_index, edge_attr = build_knn_graph(structure, node_features_np, k_neighbors)
        
        graph_data.append((node_features_np, edge_index, edge_attr))
    
    # Save to cache
    if cache_file:
        print(f"  Saving graph data to {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(graph_data, f)
    
    return graph_data


class AggregatedEmbeddingDataset(Dataset):
    """Dataset for aggregated embeddings (CNN-A)."""
    
    def __init__(self, embeddings, targets, properties=None):
        self.embeddings = torch.FloatTensor(embeddings)
        self.targets = torch.FloatTensor(targets)
        self.properties = properties
        if properties:
            self.property_tensors = {}
            for prop_name, prop_values in properties.items():
                # Convert to numpy array and handle NaN values
                prop_values = np.array(prop_values, dtype=np.float64)
                prop_values = np.nan_to_num(prop_values, nan=0.0)
                self.property_tensors[prop_name] = torch.FloatTensor(prop_values)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.properties:
            return self.embeddings[idx], self.targets[idx], {k: v[idx] for k, v in self.property_tensors.items()}
        else:
            return self.embeddings[idx], self.targets[idx]


class SequenceEmbeddingDataset(Dataset):
    """Dataset for per-atom sequence embeddings (CNN-B)."""
    
    def __init__(self, embeddings, targets, max_atoms=None, properties=None):
        """
        Args:
            embeddings: List of numpy arrays of shape [n_atoms, 256]
            targets: numpy array of targets
            max_atoms: Maximum number of atoms (for padding/truncation)
            properties: Dictionary of additional properties
        """
        self.embeddings = embeddings
        self.targets = torch.FloatTensor(targets)
        self.properties = properties
        
        # Determine max_atoms if not provided
        if max_atoms is None:
            self.max_atoms = max(emb.shape[0] for emb in embeddings)
        else:
            self.max_atoms = max_atoms
        
        if properties:
            self.property_tensors = {}
            for prop_name, prop_values in properties.items():
                # Convert to numpy array and handle NaN values
                prop_values = np.array(prop_values, dtype=np.float64)
                prop_values = np.nan_to_num(prop_values, nan=0.0)
                self.property_tensors[prop_name] = torch.FloatTensor(prop_values)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        emb = self.embeddings[idx]  # [n_atoms, 256]
        target = self.targets[idx]
        
        # Pad or truncate to max_atoms
        n_atoms = emb.shape[0]
        if n_atoms > self.max_atoms:
            # Truncate
            emb = emb[:self.max_atoms]
        elif n_atoms < self.max_atoms:
            # Pad with zeros
            padding = np.zeros((self.max_atoms - n_atoms, emb.shape[1]))
            emb = np.vstack([emb, padding])
        
        if self.properties:
            return torch.FloatTensor(emb), target, n_atoms, {k: v[idx] for k, v in self.property_tensors.items()}
        else:
            return torch.FloatTensor(emb), target, n_atoms


def collate_sequence_batch(batch):
    """Custom collate function for sequence embeddings."""
    if len(batch[0]) == 4:  # With properties
        embeddings, targets, lengths, properties = zip(*batch)
        embeddings = torch.stack(embeddings)  # [batch_size, max_atoms, 256]
        targets = torch.stack(targets)  # [batch_size]
        lengths = torch.LongTensor(lengths)  # [batch_size]
        
        # Stack properties
        stacked_properties = {}
        for prop_name in properties[0].keys():
            stacked_properties[prop_name] = torch.stack([p[prop_name] for p in properties])
        
        return embeddings, targets, lengths, stacked_properties
    else:  # Without properties
        embeddings, targets, lengths = zip(*batch)
        embeddings = torch.stack(embeddings)  # [batch_size, max_atoms, 256]
        targets = torch.stack(targets)  # [batch_size]
        lengths = torch.LongTensor(lengths)  # [batch_size]
        return embeddings, targets, lengths


if TORCH_GEOMETRIC_AVAILABLE:
    class GraphEmbeddingDataset(Dataset):
        """Dataset for graph-based embeddings (CGCNN)."""
        
        def __init__(self, graph_data, targets, properties=None):
            """
            Args:
                graph_data: List of (node_features, edge_index, edge_attr) tuples
                targets: numpy array of targets
                properties: Dictionary of additional properties (already scaled)
            """
            self.graph_data = graph_data
            self.targets = torch.FloatTensor(targets)
            self.properties = properties
            
            if properties:
                self.property_tensors = {}
                for prop_name, prop_values in properties.items():
                    # Already scaled and handled in main, just convert to tensor
                    self.property_tensors[prop_name] = torch.FloatTensor(prop_values)
        
        def __len__(self):
            return len(self.graph_data)
        
        def __getitem__(self, idx):
            node_features, edge_index, edge_attr = self.graph_data[idx]
            target = self.targets[idx]
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=torch.FloatTensor(node_features),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=target
            )
            
            # Add properties if available
            if self.properties:
                prop_list = []
                for prop_name in ['formation_energy', 'e_above_hull', 'band_gap', 
                                 'elasticity_K_VRH', 'elasticity_G_VRH']:
                    if prop_name in self.property_tensors:
                        prop_list.append(self.property_tensors[prop_name][idx].unsqueeze(0))
                
                if prop_list:
                    data.properties = torch.cat(prop_list, dim=0)
            
            return data


class CNN_A(nn.Module):
    """CNN for aggregated embeddings with kernels [3, 3, 2] and optional property concatenation."""
    
    def __init__(self, input_dim=512, num_properties=5, use_properties=True):
        super(CNN_A, self).__init__()
        
        self.use_properties = use_properties
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, padding=0)
        
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.2)
        
        # Calculate output dimension after convolutions
        # input: [batch, 1, 512]
        # after conv1: [batch, 64, 512]
        # after conv2: [batch, 32, 512]
        # after conv3: [batch, 16, 511]
        conv_output_dim = 16 * 511
        
        # First layer before concatenation
        self.fc1 = nn.Linear(conv_output_dim, 128)
        
        # Layer after property concatenation (128 + 5 properties = 133) or just 128
        fc2_input_dim = 128 + num_properties if use_properties else 128
        self.fc2 = nn.Linear(fc2_input_dim, 64)
        
        # Additional dense layer
        self.fc3 = nn.Linear(64, 32)
        
        # Final output layer
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x, properties=None):
        # x: [batch_size, 512]
        x = x.unsqueeze(1)  # [batch_size, 1, 512]
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        
        # First fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Concatenate properties after first FC layer (if enabled)
        if self.use_properties and properties is not None:
            # Stack properties in order
            prop_list = []
            for prop_name in ['formation_energy', 'e_above_hull', 'band_gap', 
                             'elasticity_K_VRH', 'elasticity_G_VRH']:
                if prop_name in properties:
                    prop_list.append(properties[prop_name].unsqueeze(1))
            
            if prop_list:
                props_tensor = torch.cat(prop_list, dim=1)  # [batch_size, num_properties]
                x = torch.cat([x, props_tensor], dim=1)  # [batch_size, 128 + num_properties]
        
        # Second fully connected layer with concatenated properties
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Third fully connected layer
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Final prediction
        thermal_pred = self.fc4(x)
        
        outputs = {'thermal_conductivity': thermal_pred.squeeze()}
        return outputs


class CNN_B(nn.Module):
    """CNN for per-atom sequence embeddings with kernels [3, 3, 2] and optional property concatenation."""
    
    def __init__(self, embedding_dim=256, num_properties=5, use_properties=True):
        super(CNN_B, self).__init__()
        
        self.use_properties = use_properties
        
        # Input: [batch, embedding_dim, n_atoms]
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, padding=0)
        
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.2)
        
        # Global pooling will be applied, so output is 32-dim
        self.fc1 = nn.Linear(32, 64)
        
        # Layer after property concatenation (64 + 5 properties = 69) or just 64
        fc2_input_dim = 64 + num_properties if use_properties else 64
        self.fc2 = nn.Linear(fc2_input_dim, 32)
        
        # Additional dense layer
        self.fc3 = nn.Linear(32, 16)
        
        # Final output layer
        self.fc4 = nn.Linear(16, 1)
    
    def forward(self, x, lengths=None, properties=None):
        # x: [batch_size, max_atoms, 256]
        x = x.transpose(1, 2)  # [batch_size, 256, max_atoms]
        
        x = self.relu(self.conv1(x))  # [batch_size, 128, max_atoms]
        x = self.relu(self.conv2(x))  # [batch_size, 64, max_atoms]
        x = self.relu(self.conv3(x))  # [batch_size, 32, max_atoms-1]
        
        # Masked global average pooling to handle variable-length sequences
        if lengths is not None:
            L = x.size(2)  # sequence length after convolutions
            device = x.device
            # Create mask: [batch_size, L] where True indicates valid positions
            # Note: after conv3 with kernel=2, padding=0, length reduces by 1
            adjusted_lengths = torch.clamp(lengths - 1, min=1)  # account for conv3 reduction
            mask = torch.arange(L, device=device).unsqueeze(0) < adjusted_lengths.unsqueeze(1)  # [B, L]
            mask = mask.float().unsqueeze(1)  # [B, 1, L]
            # Masked average: sum over valid positions and normalize by count
            x = (x * mask).sum(dim=2) / (mask.sum(dim=2) + 1e-8)  # [B, 32]
        else:
            # Fallback to regular global average pooling
            x = torch.mean(x, dim=2)  # [batch_size, 32]
        
        x = self.dropout(x)
        
        # First fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Concatenate properties after first FC layer (if enabled)
        if self.use_properties and properties is not None:
            # Stack properties in order
            prop_list = []
            for prop_name in ['formation_energy', 'e_above_hull', 'band_gap', 
                             'elasticity_K_VRH', 'elasticity_G_VRH']:
                if prop_name in properties:
                    prop_list.append(properties[prop_name].unsqueeze(1))
            
            if prop_list:
                props_tensor = torch.cat(prop_list, dim=1)  # [batch_size, num_properties]
                x = torch.cat([x, props_tensor], dim=1)  # [batch_size, 64 + num_properties]
        
        # Second fully connected layer with concatenated properties
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Third fully connected layer
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Final prediction
        thermal_pred = self.fc4(x)
        
        outputs = {'thermal_conductivity': thermal_pred.squeeze()}
        return outputs


# ============================================================================
# Graph Neural Network (CGCNN) Implementation
# ============================================================================

class WeightedLogMAELoss(nn.Module):
    """
    Weighted Log Mean Absolute Error Loss that focuses more on small log kappa values.
    Gives full weight to y_true <= 1, and decays for larger values.
    """
    
    def __init__(self, threshold=1.0, p=2, eps=1e-12):
        """
        Args:
            threshold: Value below which full weight is applied (default: 1.0)
            p: Power for weight decay (default: 2)
            eps: Small constant for numerical stability (default: 1e-12)
        """
        super(WeightedLogMAELoss, self).__init__()
        self.threshold = threshold
        self.p = p
        self.eps = eps
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Predicted log kappa values [batch_size]
            y_true: True log kappa values [batch_size]
        
        Returns:
            Weighted log MAE loss (scalar)
        """
        # Compute weights: full weight <= threshold; decay as (threshold/y)^p above threshold
        # Note: y_true and y_pred are already in log space, so we need to exponentiate
        # to get actual kappa values for weight calculation
        y_true_linear = torch.exp(y_true)  # Convert back to linear scale
        
        # Compute weights
        weights = torch.minimum(
            torch.ones_like(y_true_linear),
            (self.threshold / torch.maximum(y_true_linear, torch.tensor(self.eps, device=y_true.device))) ** self.p
        )
        
        # Compute absolute error in log space (already in log space)
        log_error = torch.abs(y_pred - y_true)
        
        # Weighted error
        weighted_error = weights * log_error
        
        # Return weighted mean
        return weighted_error.sum() / (weights.sum() + self.eps)


if TORCH_GEOMETRIC_AVAILABLE:
    class CGConv(MessagePassing):
        """Crystal Graph Convolutional layer."""
        
        def __init__(self, node_feat_dim, edge_feat_dim):
            super(CGConv, self).__init__(aggr='add')
            self.node_feat_dim = node_feat_dim
            self.edge_feat_dim = edge_feat_dim
            
            # Edge network
            self.edge_nn = nn.Sequential(
                nn.Linear(2 * node_feat_dim + edge_feat_dim, 128),
                nn.Softplus(),
                nn.Linear(128, node_feat_dim),
                nn.Softplus()
            )
            
            # Node update network
            self.node_nn = nn.Sequential(
                nn.Linear(2 * node_feat_dim, node_feat_dim),
                nn.Softplus()
            )
        
        def forward(self, x, edge_index, edge_attr):
            # x: [num_nodes, node_dim]
            # edge_index: [2, num_edges]
            # edge_attr: [num_edges, edge_dim]
            
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        def message(self, x_i, x_j, edge_attr):
            # x_i: [num_edges, node_dim] (target nodes)
            # x_j: [num_edges, node_dim] (source nodes)
            # edge_attr: [num_edges, edge_dim]
            
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
            return self.edge_nn(z)
        
        def update(self, aggr_out, x):
            # aggr_out: [num_nodes, node_dim]
            # x: [num_nodes, node_dim]
            
            z = torch.cat([x, aggr_out], dim=-1)
            return self.node_nn(z)


    class CGCNN(nn.Module):
        """Crystal Graph Convolutional Neural Network for embeddings with property concatenation."""
        
        def __init__(self, node_feat_dim=256, edge_feat_dim=1, num_conv_layers=3, num_properties=5, use_properties=True):
            super(CGCNN, self).__init__()
            
            self.use_properties = use_properties
            self.node_feat_dim = node_feat_dim
            
            # Initial node embedding projection (if needed)
            self.node_embedding = nn.Linear(node_feat_dim, node_feat_dim)
            
            # Graph convolutional layers
            self.conv_layers = nn.ModuleList([
                CGConv(node_feat_dim, edge_feat_dim) for _ in range(num_conv_layers)
            ])
            
            # Batch normalization layers
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(node_feat_dim) for _ in range(num_conv_layers)
            ])
            
            # Fully connected layers after pooling
            self.fc1 = nn.Linear(node_feat_dim * 2, 128)  # *2 for mean + max pooling
            
            # Layer after property concatenation
            fc2_input_dim = 128 + num_properties if use_properties else 128
            self.fc2 = nn.Linear(fc2_input_dim, 64)
            
            # Additional dense layers
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, data):
            # data is a PyTorch Geometric Data or Batch object
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            properties = data.properties if hasattr(data, 'properties') else None
            
            # Initial node embedding
            x = self.relu(self.node_embedding(x))
            
            # Graph convolutions
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x_new = conv(x, edge_index, edge_attr)
                x_new = bn(x_new)
                x = self.relu(x_new) + x  # Residual connection
            
            # Global pooling (mean + max)
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
            
            # Fully connected layers
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            
            # Concatenate properties after first FC layer (if enabled)
            if self.use_properties and properties is not None:
                x = torch.cat([x, properties], dim=1)
            
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            
            x = self.relu(self.fc3(x))
            x = self.dropout(x)
            
            thermal_pred = self.fc4(x)
            
            outputs = {'thermal_conductivity': thermal_pred.squeeze()}
            return outputs


def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, use_weighted_loss=False):
    """
    Train a model with property concatenation and return predictions.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device (cpu or cuda)
        epochs: Number of training epochs
        lr: Learning rate
        use_weighted_loss: If True, use WeightedLogMAELoss; if False, use MSELoss
    """
    model = model.to(device)
    
    # Choose loss function
    if use_weighted_loss:
        criterion = WeightedLogMAELoss(threshold=1.0, p=2, eps=1e-12)
        print(f"    Using WeightedLogMAELoss (focusing on log(κ) < 1)")
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 4:  # Sequence data with properties
                inputs, targets, lengths, properties = batch
                inputs, targets = inputs.to(device), targets.to(device)
                lengths = lengths.to(device)
                properties = {k: v.to(device) for k, v in properties.items()}
                outputs = model(inputs, lengths, properties)
            elif len(batch) == 3 and isinstance(batch[2], dict):  # Aggregated data with properties
                inputs, targets, properties = batch
                inputs, targets = inputs.to(device), targets.to(device)
                properties = {k: v.to(device) for k, v in properties.items()}
                outputs = model(inputs, properties)
            elif len(batch) == 3:  # Sequence data without properties
                inputs, targets, lengths = batch
                inputs, targets = inputs.to(device), targets.to(device)
                lengths = lengths.to(device)
                outputs = model(inputs, lengths)
            else:  # Aggregated data without properties
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            
            # Calculate loss (only thermal conductivity)
            loss = criterion(outputs['thermal_conductivity'], targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:  # Sequence data with properties
                    inputs, targets, lengths, properties = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    lengths = lengths.to(device)
                    properties = {k: v.to(device) for k, v in properties.items()}
                    outputs = model(inputs, lengths, properties)
                elif len(batch) == 3 and isinstance(batch[2], dict):  # Aggregated data with properties
                    inputs, targets, properties = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    properties = {k: v.to(device) for k, v in properties.items()}
                    outputs = model(inputs, properties)
                elif len(batch) == 3:  # Sequence data without properties
                    inputs, targets, lengths = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    lengths = lengths.to(device)
                    outputs = model(inputs, lengths)
                else:  # Aggregated data without properties
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                
                # Calculate validation loss (only thermal conductivity)
                loss = criterion(outputs['thermal_conductivity'], targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) <= 5:
            print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and targets."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:  # Sequence data with properties
                inputs, targets, lengths, properties = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                properties = {k: v.to(device) for k, v in properties.items()}
                outputs = model(inputs, lengths, properties)
            elif len(batch) == 3 and isinstance(batch[2], dict):  # Aggregated data with properties
                inputs, targets, properties = batch
                inputs = inputs.to(device)
                properties = {k: v.to(device) for k, v in properties.items()}
                outputs = model(inputs, properties)
            elif len(batch) == 3:  # Sequence data without properties
                inputs, targets, lengths = batch
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                outputs = model(inputs, lengths)
            else:  # Aggregated data without properties
                inputs, targets = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
            
            all_preds.extend(outputs['thermal_conductivity'].cpu().numpy())
            all_targets.extend(targets.numpy())
    
    return np.array(all_preds), np.array(all_targets)


if TORCH_GEOMETRIC_AVAILABLE:
    def train_graph_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3):
        """Train a graph model (CGCNN) and return the trained model."""
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = model.state_dict()
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs['thermal_conductivity'], batch.y)
                loss.backward()
                # Gradient clipping to stabilize training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    loss = criterion(outputs['thermal_conductivity'], batch.y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Print progress
            if epoch < 5 or (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model


    def evaluate_graph_model(model, test_loader, device):
        """Evaluate graph model and return predictions and targets."""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                all_preds.extend(outputs['thermal_conductivity'].cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        return np.array(all_preds), np.array(all_targets)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train CNN models on MLIP embeddings for thermal conductivity prediction')
    parser.add_argument('--use-properties', action='store_true', default=False,
                        help='Use material properties as additional features (default: False)')
    parser.add_argument('--model', type=str, choices=['CNN-A', 'CNN-B', 'CGCNN', 'TabPFN', 'all'], default='all',
                        help='Which model to train: CNN-A (aggregated), CNN-B (sequence), CGCNN (graph), TabPFN (tabular), or all (default: all)')
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['ood_split', 'random_split', 'space_group_split'],
                        help='Which splits to process (default: all three)')
    parser.add_argument('--ood-scaler', type=str, choices=['minmax', 'standard', 'robust'], default='robust',
                        help='Scaler type for OOD split (default: robust, handles outliers better)')
    parser.add_argument('--weighted-loss', action='store_true', default=False,
                        help='Use weighted loss that focuses more on log(κ) < 1 (default: False)')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("CONFIGURATION")
    print('='*80)
    print(f"Use material properties: {args.use_properties}")
    print(f"Model(s) to train: {args.model}")
    print(f"Splits to process: {args.splits}")
    print(f"OOD scaler: {args.ood_scaler}")
    print(f"Weighted loss (focus on log(κ) < 1): {args.weighted_loss}")
    print('='*80)
    
    # Setup device with CUDA compatibility check
    def test_cuda_compatibility():
        """Test if CUDA is available and compatible."""
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        try:
            # Test basic CUDA operations
            device = torch.device("cuda")
            x = torch.randn(10, 10).to(device)
            y = torch.arange(100, device=device)
            _ = x @ x.T
            _ = y.sum()
            return True, "CUDA compatible"
        except Exception as e:
            return False, f"CUDA error: {str(e)[:100]}..."

    cuda_ok, cuda_msg = test_cuda_compatibility()

    if cuda_ok:
        device = torch.device("cuda")
        print(f"\nUsing device: {device}")
    else:
        device = torch.device("cpu")
        print(f"\nCUDA compatibility issue: {cuda_msg}")
        print(f"Using device: {device} (CPU fallback)")
    
    # Load ORB model
    print("\nLoading orb-v3-conservative-20-omat model...")
    try:
        orbff = pretrained.orb_v3_conservative_20_omat(
            device=device,
            precision="float32-highest",
            compile=False  # Disable compilation to avoid C++ compiler requirement
        )
        print("Model loaded successfully!")
    except Exception as e:
        if "cuda" in str(e).lower():
            print(f"CUDA error loading model: {e}")
            print("Attempting to load on CPU...")
            device = torch.device("cpu")
            orbff = pretrained.orb_v3_conservative_20_omat(
                device=device,
                precision="float32-highest",
                compile=False
            )
            print("Model loaded successfully on CPU!")
        else:
            raise e

    orbff.eval()
    
    # Data splits to process
    splits = args.splits
    
    # Store results
    results = {}
    if args.model in ['CNN-A', 'all']:
        results['CNN-A'] = {}
    if args.model in ['CNN-B', 'all']:
        results['CNN-B'] = {}
    if args.model in ['CGCNN', 'all'] and TORCH_GEOMETRIC_AVAILABLE:
        results['CGCNN'] = {}
    if args.model in ['TabPFN', 'all'] and TABPFN_AVAILABLE:
        results['TabPFN'] = {}
    
    # Process each split
    for split_name in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split_name}")
        print('='*60)
        
        # Load data
        train_structures, test_structures, y_train, y_test, train_props, test_props = load_split_data(split_name)
        
        # Normalize properties (use different scaler for OOD to handle extrapolation better)
        if args.use_properties:
            if split_name == 'ood_split':
                if args.ood_scaler == 'robust':
                    property_scaler = RobustScaler()  # Less sensitive to outliers
                    print("\n  Normalizing material properties with RobustScaler (for OOD extrapolation)...")
                elif args.ood_scaler == 'standard':
                    property_scaler = StandardScaler()  # Standard normalization
                    print("\n  Normalizing material properties with StandardScaler...")
                else:
                    property_scaler = MinMaxScaler(feature_range=(-1, 1))  # Wider range for OOD
                    print("\n  Normalizing material properties with MinMaxScaler (range: -1 to 1)...")
            else:
                property_scaler = MinMaxScaler()
                print("\n  Normalizing material properties with MinMaxScaler...")
            
            # Combine all properties into matrices
            train_prop_matrix = np.column_stack([
                np.array(train_props['formation_energy'], dtype=np.float64),
                np.array(train_props['e_above_hull'], dtype=np.float64),
                np.array(train_props['band_gap'], dtype=np.float64),
                np.array(train_props['elasticity_K_VRH'], dtype=np.float64),
                np.array(train_props['elasticity_G_VRH'], dtype=np.float64)
            ])
            
            test_prop_matrix = np.column_stack([
                np.array(test_props['formation_energy'], dtype=np.float64),
                np.array(test_props['e_above_hull'], dtype=np.float64),
                np.array(test_props['band_gap'], dtype=np.float64),
                np.array(test_props['elasticity_K_VRH'], dtype=np.float64),
                np.array(test_props['elasticity_G_VRH'], dtype=np.float64)
            ])
            
            # Handle NaN values before scaling
            train_prop_matrix = np.nan_to_num(train_prop_matrix, nan=0.0)
            test_prop_matrix = np.nan_to_num(test_prop_matrix, nan=0.0)
            
            # Fit scaler on training data and transform both sets
            train_prop_matrix_scaled = property_scaler.fit_transform(train_prop_matrix)
            test_prop_matrix_scaled = property_scaler.transform(test_prop_matrix)
            
            # Convert back to dictionaries
            prop_names = ['formation_energy', 'e_above_hull', 'band_gap', 'elasticity_K_VRH', 'elasticity_G_VRH']
            train_props_scaled = {name: train_prop_matrix_scaled[:, i] for i, name in enumerate(prop_names)}
            test_props_scaled = {name: test_prop_matrix_scaled[:, i] for i, name in enumerate(prop_names)}
        else:
            # Create dummy scaled properties (won't be used)
            train_props_scaled = None
            test_props_scaled = None
        
        # Extract embeddings for CNN-A (aggregated) with caching
        print("\nCNN-A: Aggregated embeddings")
        train_cache_file = f"cache/{split_name}_train_agg_embeddings.pkl"
        test_cache_file = f"cache/{split_name}_test_agg_embeddings.pkl"
        train_emb_agg = extract_embeddings(train_structures, orbff, device, return_sequences=False, cache_file=train_cache_file)
        test_emb_agg = extract_embeddings(test_structures, orbff, device, return_sequences=False, cache_file=test_cache_file)
        
        # Extract embeddings for CNN-B (sequences) with caching
        print("\nCNN-B: Sequence embeddings")
        train_seq_cache_file = f"cache/{split_name}_train_seq_embeddings.pkl"
        test_seq_cache_file = f"cache/{split_name}_test_seq_embeddings.pkl"
        train_emb_seq = extract_embeddings(train_structures, orbff, device, return_sequences=True, cache_file=train_seq_cache_file)
        test_emb_seq = extract_embeddings(test_structures, orbff, device, return_sequences=True, cache_file=test_seq_cache_file)
        
        # Determine max_atoms for consistent padding across train and test
        max_atoms = max(max(emb.shape[0] for emb in train_emb_seq),
                       max(emb.shape[0] for emb in test_emb_seq))
        print(f"  Max atoms across all structures: {max_atoms}")
        
        # Create datasets and dataloaders for CNN-A with scaled properties
        train_dataset_a = AggregatedEmbeddingDataset(train_emb_agg, y_train, properties=train_props_scaled)
        test_dataset_a = AggregatedEmbeddingDataset(test_emb_agg, y_test, properties=test_props_scaled)
        
        train_loader_a = DataLoader(train_dataset_a, batch_size=32, shuffle=True)
        test_loader_a = DataLoader(test_dataset_a, batch_size=32, shuffle=False)
        
        # Create datasets and dataloaders for CNN-B with scaled properties
        train_dataset_b = SequenceEmbeddingDataset(train_emb_seq, y_train, max_atoms=max_atoms, properties=train_props_scaled)
        test_dataset_b = SequenceEmbeddingDataset(test_emb_seq, y_test, max_atoms=max_atoms, properties=test_props_scaled)
        
        train_loader_b = DataLoader(train_dataset_b, batch_size=32, shuffle=True, 
                                    collate_fn=collate_sequence_batch)
        test_loader_b = DataLoader(test_dataset_b, batch_size=32, shuffle=False,
                                   collate_fn=collate_sequence_batch)
        
        # Determine epochs (10 for OOD, 100 for others)
        num_epochs = 100 if split_name == 'ood_split' else 100
        
        # Train CNN-A if requested
        if args.model in ['CNN-A', 'all']:
            prop_str = "with properties" if args.use_properties else "without properties"
            print(f"\n  Training CNN-A (aggregated embeddings {prop_str}, epochs={num_epochs})...")
            model_a = CNN_A(input_dim=512, use_properties=args.use_properties)
            model_a = train_model(model_a, train_loader_a, test_loader_a, device, 
                                 epochs=num_epochs, lr=8e-5, use_weighted_loss=args.weighted_loss)
            
            # Evaluate CNN-A
            preds_a, targets_a = evaluate_model(model_a, test_loader_a, device)
            r2_a = r2_score(targets_a, preds_a)
            mae_a = mean_absolute_error(targets_a, preds_a)
            
            results['CNN-A'][split_name] = {
                'predictions': preds_a,
                'targets': targets_a,
                'r2': r2_a,
                'mae': mae_a
            }
            
            print(f"  CNN-A Results - R2: {r2_a:.4f}, MAE: {mae_a:.4f}")
        
        # Train CNN-B if requested
        if args.model in ['CNN-B', 'all']:
            prop_str = "with properties" if args.use_properties else "without properties"
            print(f"\n  Training CNN-B (sequence embeddings {prop_str}, epochs={num_epochs})...")
            model_b = CNN_B(embedding_dim=256, use_properties=args.use_properties)
            model_b = train_model(model_b, train_loader_b, test_loader_b, device, 
                                 epochs=num_epochs, lr=8e-5, use_weighted_loss=args.weighted_loss)
            
            # Evaluate CNN-B
            preds_b, targets_b = evaluate_model(model_b, test_loader_b, device)
            r2_b = r2_score(targets_b, preds_b)
            mae_b = mean_absolute_error(targets_b, preds_b)
            
            results['CNN-B'][split_name] = {
                'predictions': preds_b,
                'targets': targets_b,
                'r2': r2_b,
                'mae': mae_b
            }
            
            print(f"  CNN-B Results - R2: {r2_b:.4f}, MAE: {mae_b:.4f}")
        
        # Train CGCNN if requested
        if args.model in ['CGCNN', 'all'] and TORCH_GEOMETRIC_AVAILABLE:
            # Extract graph data with caching
            print("\nCGCNN: Graph data using ORB's native graph structure")
            train_graph_cache_file = f"cache/{split_name}_train_graph_data.pkl"
            test_graph_cache_file = f"cache/{split_name}_test_graph_data.pkl"
            train_graph_data = extract_graph_data(train_structures, orbff, device, k_neighbors=12, cache_file=train_graph_cache_file)
            test_graph_data = extract_graph_data(test_structures, orbff, device, k_neighbors=12, cache_file=test_graph_cache_file)
            
            # Create datasets and dataloaders for CGCNN
            train_dataset_cgcnn = GraphEmbeddingDataset(train_graph_data, y_train, properties=train_props_scaled)
            test_dataset_cgcnn = GraphEmbeddingDataset(test_graph_data, y_test, properties=test_props_scaled)
            
            # Custom collate function to properly batch properties
            def collate_graph_batch(batch_list):
                batch = Batch.from_data_list(batch_list)
                # Extract and stack properties if they exist
                if hasattr(batch_list[0], 'properties'):
                    properties = torch.stack([data.properties for data in batch_list])
                    batch.properties = properties
                return batch
            
            train_loader_cgcnn = DataLoader(train_dataset_cgcnn, batch_size=32, shuffle=True, 
                                           collate_fn=collate_graph_batch)
            test_loader_cgcnn = DataLoader(test_dataset_cgcnn, batch_size=32, shuffle=False,
                                          collate_fn=collate_graph_batch)
            
            prop_str = "with properties" if args.use_properties else "without properties"
            print(f"\n  Training CGCNN (using ORB native edges {prop_str}, epochs={num_epochs})...")
            model_cgcnn = CGCNN(node_feat_dim=256, edge_feat_dim=1, use_properties=args.use_properties)
            model_cgcnn = train_graph_model(model_cgcnn, train_loader_cgcnn, test_loader_cgcnn, device, epochs=num_epochs, lr=1e-3)
            
            # Evaluate CGCNN
            preds_cgcnn, targets_cgcnn = evaluate_graph_model(model_cgcnn, test_loader_cgcnn, device)
            r2_cgcnn = r2_score(targets_cgcnn, preds_cgcnn)
            mae_cgcnn = mean_absolute_error(targets_cgcnn, preds_cgcnn)
            
            results['CGCNN'][split_name] = {
                'predictions': preds_cgcnn,
                'targets': targets_cgcnn,
                'r2': r2_cgcnn,
                'mae': mae_cgcnn
            }
            
            print(f"  CGCNN Results - R2: {r2_cgcnn:.4f}, MAE: {mae_cgcnn:.4f}")
        elif args.model in ['CGCNN', 'all'] and not TORCH_GEOMETRIC_AVAILABLE:
            print("\n  Warning: CGCNN requested but PyTorch Geometric is not available. Skipping CGCNN training.")
        
        # Train TabPFN if requested
        if args.model in ['TabPFN', 'all'] and TABPFN_AVAILABLE:
            print("\nTabPFN: Tabular transformer using masked statistics (mean, std, max)")
            
            # Extract masked statistics from sequence embeddings
            print("  Extracting masked statistics from node embeddings...")
            train_lengths = [emb.shape[0] for emb in train_emb_seq]
            test_lengths = [emb.shape[0] for emb in test_emb_seq]
            
            train_graph_features = extract_masked_statistics(train_emb_seq, train_lengths)  # [n_train, 768]
            test_graph_features = extract_masked_statistics(test_emb_seq, test_lengths)  # [n_test, 768]
            
            print(f"  Masked statistics shape: {train_graph_features.shape} (mean+std+max pooling)")
            
            # Prepare tabular input: concatenate graph features with properties if enabled
            if args.use_properties and train_props_scaled is not None:
                # Convert properties dict to array
                prop_names = ['formation_energy', 'e_above_hull', 'band_gap', 'elasticity_K_VRH', 'elasticity_G_VRH']
                train_props_array = np.column_stack([train_props_scaled[name] for name in prop_names if name in train_props_scaled])
                test_props_array = np.column_stack([test_props_scaled[name] for name in prop_names if name in test_props_scaled])
                
                # Concatenate graph features with properties
                X_train_tabular = np.concatenate([train_graph_features, train_props_array], axis=1)
                X_test_tabular = np.concatenate([test_graph_features, test_props_array], axis=1)
                prop_str = "with properties"
            else:
                X_train_tabular = train_graph_features
                X_test_tabular = test_graph_features
                prop_str = "without properties"
            
            print(f"\n  Training TabPFN ({prop_str})...")
            print(f"  Input shape: {X_train_tabular.shape}")
            
            # Detect GPU availability
            tabpfn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"  TabPFN will run on: {tabpfn_device.upper()}")
            
            # Initialize and train TabPFN
            # TabPFN limitations: max 100 features (can go up to 500 with some versions), max 10000 samples on GPU
            pca_threshold = 60  # Only apply PCA if features exceed this threshold
            if X_train_tabular.shape[1] > pca_threshold:
                print(f"  Warning: TabPFN feature limit exceeded. Current: {X_train_tabular.shape[1]}")
                print(f"  Using PCA to reduce dimensionality to {pca_threshold}...")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=pca_threshold, random_state=RANDOM_SEED)
                X_train_tabular = pca.fit_transform(X_train_tabular)
                X_test_tabular = pca.transform(X_test_tabular)
                print(f"  Reduced input shape: {X_train_tabular.shape}")
                print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
            
            # TabPFN sample limitations: max 1000 on CPU, max 10000 on GPU
            max_samples = 10000 if tabpfn_device == 'cuda' else 1000
            if X_train_tabular.shape[0] > max_samples:
                print(f"  Warning: TabPFN on {tabpfn_device.upper()} supports max {max_samples} training samples. Current: {X_train_tabular.shape[0]}")
                print(f"  Randomly sampling {max_samples} samples...")
                np.random.seed(RANDOM_SEED)
                sample_idx = np.random.choice(X_train_tabular.shape[0], max_samples, replace=False)
                X_train_tabular = X_train_tabular[sample_idx]
                y_train_sample = y_train[sample_idx]
            else:
                y_train_sample = y_train
            
            try:
                # Initialize TabPFN with GPU if available
                tabpfn = TabPFNRegressor(device=tabpfn_device)
                tabpfn.fit(X_train_tabular, y_train_sample)
                
                # Predict on test set
                preds_tabpfn = tabpfn.predict(X_test_tabular)
                r2_tabpfn = r2_score(y_test, preds_tabpfn)
                mae_tabpfn = mean_absolute_error(y_test, preds_tabpfn)
                
                results['TabPFN'][split_name] = {
                    'predictions': preds_tabpfn,
                    'targets': y_test,
                    'r2': r2_tabpfn,
                    'mae': mae_tabpfn
                }
                
                print(f"  TabPFN Results - R2: {r2_tabpfn:.4f}, MAE: {mae_tabpfn:.4f}")
            except Exception as e:
                print(f"  Error training TabPFN: {e}")
                print(f"  Skipping TabPFN for {split_name}")
        elif args.model in ['TabPFN', 'all'] and not TABPFN_AVAILABLE:
            print("\n  Warning: TabPFN requested but not available. Skipping TabPFN training.")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Split':<25} {'Model':<10} {'R2 Score':<15} {'MAE':<15}")
    print("-"*80)
    for split_name in splits:
        for model_name in results.keys():
            if split_name in results[model_name]:
                r2 = results[model_name][split_name]['r2']
                mae = results[model_name][split_name]['mae']
                print(f"{split_name:<25} {model_name:<10} {r2:<15.4f} {mae:<15.4f}")
    print("="*80)
    
    # Create visualization
    print("\nCreating scatter plots...")
    
    # Determine subplot layout based on models to plot
    model_names = list(results.keys())
    n_models = len(model_names)
    n_splits = len(splits)
    
    fig, axes = plt.subplots(n_splits, n_models, figsize=(6*n_models, 5*n_splits))
    
    # Handle single subplot case
    if n_splits == 1 and n_models == 1:
        axes = [[axes]]
    elif n_splits == 1:
        axes = [axes]
    elif n_models == 1:
        axes = [[ax] for ax in axes]
    
    fig.suptitle('Log Thermal Conductivity Predictions', fontsize=16, fontweight='bold')
    
    for i, split_name in enumerate(splits):
        for j, model_name in enumerate(model_names):
            ax = axes[i][j]
            
            if split_name in results[model_name]:
                preds = results[model_name][split_name]['predictions']
                targets = results[model_name][split_name]['targets']
                r2 = results[model_name][split_name]['r2']
                mae = results[model_name][split_name]['mae']
            else:
                # Skip if model wasn't trained on this split
                ax.axis('off')
                continue
            
            # Scatter plot
            ax.scatter(targets, preds, alpha=0.5, s=20)
            
            # Diagonal line (perfect prediction)
            min_val = min(targets.min(), preds.min())
            max_val = max(targets.max(), preds.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            # Labels and title
            ax.set_xlabel('True log(κ_lat)', fontsize=10)
            ax.set_ylabel('Predicted log(κ_lat)', fontsize=10)
            
            split_display = split_name.replace('_', ' ').title()
            ax.set_title(f'{split_display} - {model_name}', fontsize=11, fontweight='bold')
            
            # Add R2 and MAE as text
            textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f}'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermal_conductivity_predictions.png', dpi=300, bbox_inches='tight')
    print("Scatter plots saved to 'thermal_conductivity_predictions.png'")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

