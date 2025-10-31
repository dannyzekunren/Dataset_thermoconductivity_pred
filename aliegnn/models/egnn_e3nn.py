"""E3NN-Enhanced EGNN with Spherical Harmonics for Equivariant Node Features"""
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
import math
from typing import Dict, List, Tuple
from aliegnn.models.utils import RBFExpansion
from aliegnn.utils import BaseSettings

try:
    import e3nn
    from e3nn import o3
    from e3nn.nn import Gate, NormActivation
    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False
    print("Warning: e3nn not installed. Please install with: pip install e3nn")


class SphericalHarmonicsEmbedding(nn.Module):
    
    def __init__(
        self, 
        max_l: int = 2, 
        num_channels: int = 16,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        use_norm: bool = True
    ):
        super().__init__()
        if not HAS_E3NN:
            raise ImportError("e3nn is required for this module")
            
        self.max_l = max_l
        self.num_channels = num_channels
        self.use_norm = use_norm
        
        self.spherical_harmonics = o3.SphericalHarmonics(
            list(range(max_l + 1)), 
            normalize=True, 
            normalization='component'
        )
        
        self.rbf = RBFExpansion(vmin=vmin, vmax=vmax, bins=bins)
        self.bins = bins
        
        irreps_radial = o3.Irreps(f"{bins}x0e")
        
        irreps_angular = o3.Irreps.spherical_harmonics(lmax=self.max_l)
        
        irreps_out_list = []
        for l in range(self.max_l + 1):
            parity = (-1) ** l
            irreps_out_list.append((self.num_channels, (l, parity)))
        self.irreps_sh = o3.Irreps(irreps_out_list)
        
        self.tensor_product = o3.TensorProduct(
            irreps_radial, 
            irreps_angular, 
            self.irreps_sh,
            [
                (i, j, k, "uvw", True)  
                for i, (mul_in1, ir_in1) in enumerate(irreps_radial)
                for j, (mul_in2, ir_in2) in enumerate(irreps_angular)
                for k, (mul_out, ir_out) in enumerate(self.irreps_sh)
                if ir_out in ir_in1 * ir_in2
            ],
            shared_weights=False,   
            internal_weights=False,   
        )
        
        radial_MLP = [64, 64]  
        self.conv_tp_weights = nn.Sequential(
            nn.Linear(bins, radial_MLP[0]),
            nn.SiLU(),
            nn.Linear(radial_MLP[0], radial_MLP[1]),
            nn.SiLU(),
            nn.Linear(radial_MLP[1], self.tensor_product.weight_numel)
        )
        
        # Normalization for invariant features
        if self.use_norm:
            invariant_dim = self.num_channels * (self.max_l + 1)
            self.invariant_norm = nn.LayerNorm(invariant_dim)
        
    def get_total_channels(self):
        return self.irreps_sh.dim
    
    def get_invariant_channels(self):
        return self.num_channels * (self.max_l + 1)
    
    def extract_invariants(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: [E, total_channels] spherical harmonics features
            
        Returns:
            [E, invariant_channels] 
        """
        invariants = []
        offset = 0
        
        for l in range(self.max_l + 1):
            mul = self.num_channels
            dim = mul * (2*l + 1)
            chunk = edge_features[:, offset:offset+dim]
            
            if l == 0:
                invariants.append(chunk)  # [E, num_channels]
            else:
                # reshape to [E, num_channels, 2*l+1]
                chunk_reshaped = chunk.view(-1, mul, 2*l+1)
                norms = torch.norm(chunk_reshaped, dim=2)
                invariants.append(norms)
            
            offset += dim
        
        #  [E, num_channels * (max_l + 1)]
        invariants_concat = torch.cat(invariants, dim=1)
        
        if self.use_norm:
            invariants_concat = self.invariant_norm(invariants_concat)
        
        return invariants_concat
        
    def forward(self, edge_vec: torch.Tensor, edge_length: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_vec: [E, 3]  (x_i - x_j)
            edge_length: [E, 1]  ||x_i - x_j||
            
        Returns:
            [E, total_channels] 
        """
        # Y_l^m(edge_vec/||edge_vec||)
        edge_sh = self.spherical_harmonics(edge_vec)  # shape [E, sum(2l+1)]
        
        radial_basis = self.rbf(edge_length.squeeze(-1))  # shape [E, bins]
        
        tp_weights = self.conv_tp_weights(radial_basis)  # shape [E, weight_numel]
        
        edge_features = self.tensor_product(radial_basis, edge_sh, tp_weights)  # shape [E, total_channels]

        return edge_features

class Readout(nn.Module):
    def __init__(self, sh_embedding, hidden_features):
        super().__init__()
        self.readout = o3.Linear(
            irreps_in=sh_embedding.irreps_sh,
            irreps_out=o3.Irreps(f"{hidden_features}x0e")
        )

        invariant_dim = sh_embedding.num_channels * (sh_embedding.max_l + 1)
        #self.norm_proj = nn.Linear(invariant_dim, hidden_features)
        #self.combine = nn.Linear(hidden_features * 2, hidden_features)
        
    def forward(self, edge_features, sh_embedding):

        learnable_out = self.readout(edge_features)

        #norm_out = self.norm_proj(sh_embedding.extract_invariants(edge_features))
        #return self.combine(torch.cat([learnable_out, norm_out], dim=1))
        return learnable_out
class EquivariantLinear(nn.Module):
    
    def __init__(self, irreps_in: str, irreps_out: str):
        super().__init__()
        if not HAS_E3NN:
            raise ImportError("e3nn is required for this module")
            
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        
        self.linear = o3.Linear(self.irreps_in, self.irreps_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class E3NNConv(nn.Module):
    
    def __init__(
        self, 
        input_features: int,
        hidden_features: int, 
        output_features: int,
        max_l: int = 2,
        num_radial_channels: int = 16,
        use_attention: bool = False,
        use_sh_norm: bool = True,  
    ):
        super().__init__()
        if not HAS_E3NN:
            raise ImportError("e3nn is required for this module")
            
        self.in_size = input_features
        self.hidden_size = hidden_features
        self.out_size = output_features
        self.max_l = max_l
        self.num_radial_channels = num_radial_channels
        self.edge_feat_size = hidden_features
        self.use_attention = use_attention

        self.sh_embedding = SphericalHarmonicsEmbedding(
            self.max_l, 
            self.num_radial_channels,
            use_norm=use_sh_norm
        )
        
        self.use_learnable_readout = True  
        
        if self.use_learnable_readout:
            self.readout = Readout(self.sh_embedding, self.hidden_size)
            sh_invariant_dim = self.hidden_size
        else:
            sh_invariant_dim = self.sh_embedding.get_invariant_channels()
        
        if self.edge_feat_size is not None:
            edge_input_dim = self.hidden_size*2 + sh_invariant_dim + self.hidden_size
        else:
            edge_input_dim = self.hidden_size*2 + sh_invariant_dim
        self.edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.node_network = nn.Sequential(
            nn.Linear(self.in_size + self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.out_size)
        )
        
        self.coord_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1, bias=False)
        )

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.atten_drop = nn.Dropout(0.1)

    def message(self, edges):
        """message function for E3NN"""
        if edges.data.get("a") is not None:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["sh_invariant"],  
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["sh_invariant"]], dim=-1
            )
        msg_h = self.edge_network(f)  # [E, hidden_size]
        msg_x = self.coord_network(msg_h) * edges.data['x_vec']  # [E, 3]
        
        if self.use_attention:
            attention = self.leaky_relu(msg_h)
            return {"msg_x": msg_x, "msg_h": msg_h, "attention_logits": attention}
        else:
            return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(
        self, 
        graph: dgl.DGLGraph, 
        node_feat: torch.Tensor, 
        coord_feat: torch.Tensor, 
        edge_feat: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph: DGL graph
            node_feat: [N, in_size]  node features
            coord_feat: [N, 3] node coordinates
            edge_feat: [E, edge_feat_size] edge features (optional)
            
        Returns:
            updated_node_feat: [N, out_size] updated node features
            updated_coord_feat: [N, 3] updated coordinates
        """
        graph = graph.local_var()
        with graph.local_scope():
            # node features
            graph.ndata["h"] = node_feat
            # coordinate features
            graph.ndata["x"] = coord_feat
            # edge features
            if edge_feat is not None:
                graph.edata["a"] = edge_feat
                
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))  # x_i - x_j
            edge_dist = torch.norm(graph.edata["x_diff"], dim=1, keepdim=True)
            graph.edata["x_vec"] = graph.edata["x_diff"]/(edge_dist + 1e-15)  
            edge_vec = graph.edata["x_vec"]

            sh_features = self.sh_embedding(edge_vec, edge_dist)  # [E, 144]
            
            if self.use_learnable_readout:
                sh_invariant = self.readout(sh_features, self.sh_embedding)  # [E, hidden_size]
            else:
                sh_invariant = self.sh_embedding.extract_invariants(sh_features)  # [E, 48]
            
            graph.edata["sh_invariant"] = sh_invariant
            graph.apply_edges(self.message)
            
            if self.use_attention:
                graph.edata["attention"] = self.atten_drop(
                    dgl.ops.edge_softmax(graph, graph.edata["attention_logits"])
                )
            else:
                graph.edata["attention"] = torch.ones(
                    (graph.number_of_edges(), self.hidden_size), 
                    device=node_feat.device
                )

            if edge_feat is not None:
                updated_edge_feat = edge_feat + graph.edata["msg_h"]
            
            graph.edata["weighted_msg_h"] = graph.edata["msg_h"] * graph.edata["attention"]
            graph.update_all(
                fn.copy_e("weighted_msg_h", "m"),  
                fn.sum("m", "aggregated_message")   
            )
            
            aggregated = graph.ndata["aggregated_message"]
            updated_node_feat = self.node_network(torch.cat([node_feat, aggregated], dim=-1))
            
            graph.update_all(
                fn.copy_e("msg_x", "m"),       
                fn.mean("m", "coord_delta")
            )
            
            updated_coord_feat = coord_feat + graph.ndata["coord_delta"]
            
            if edge_feat is not None:
                return updated_node_feat, updated_coord_feat, updated_edge_feat
            else:
                return updated_node_feat, updated_coord_feat
