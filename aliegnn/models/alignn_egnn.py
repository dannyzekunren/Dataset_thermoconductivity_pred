"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union
import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
from aliegnn.models.egnnconv import EGNNConv
from aliegnn.models.egnnconv_GATv2 import EGNNConv as EGNNConvGATv2
from aliegnn.models.egnn_e3nn import E3NNConv
#from dgl.nn.pytorch.conv import EGNNConv

# from dgl.nn.functional import edge_softmax
from typing import Literal
from torch import nn
from torch.nn import functional as F

from aliegnn.models.utils import RBFExpansion
from aliegnn.utils import BaseSettings


class ALIEGNNConfig(BaseSettings):

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # E3NN spherical harmonics options
    max_l: int = 2  # 最大角量子数
    num_radial_channels: int = 16  # 每个l的径向通道数

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIEGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y

class EGNNGATv2Conv(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
    ):
        """Initialize parameters for ALIEGNN update."""
        super().__init__()
        self.residual = True
        self.node_update = EGNNConvGATv2(
            input_features,
            hidden_features,
            output_features,
            hidden_features,
        )
        self.edge_update = EdgeGatedGraphConv(
            input_features,
            output_features
        )
    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        coord: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """
        Node, edge, and coord updates for EGNN layer.
        x: node input features
        y: edge input features
        coord: node coordinates
        z: edge pair input features
        """  
        g = g.local_var()
        lg = lg.local_var()

       # Edge-gated graph convolution update on crystal graph
        x, coord, m, _ = self.node_update(g, x, coord, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, coord, y, z
        
class EGNNGraphConv_line(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
    ):
        """Initialize parameters for ALIEGNN update."""
        super().__init__()
        self.residual = True
        self.node_update = EGNNConv(
            input_features,
            hidden_features,
            output_features,
            hidden_features,
        )
        '''
        self.edge_update = EdgeGatedGraphConv(
            input_features,
            output_features
        )
        '''
        self.edge_update = EGNNConv(
            input_features,
            hidden_features,
            output_features,
            hidden_features,
        )
    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        coord: torch.Tensor,
        y: torch.Tensor,
        vector: torch.Tensor,
        z: torch.Tensor,
    ):
        """
        Node, edge, and coord updates for EGNN layer.
        x: node input features
        y: edge input features
        coord: node coordinates
        z: edge pair input features
        """  
        g = g.local_var()
        lg = lg.local_var()

       # Edge-gated graph convolution update on crystal graph
        x, coord, m, _ = self.node_update(g, x, coord, y)

        # Edge-gated graph convolution update on crystal graph
        y, vector, z, _ = self.edge_update(lg, m, vector, z)

        return x, coord, y, vector, z

class E3NNGraphConv(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
    ):
        """Initialize parameters for ALIEGNN update."""
        super().__init__()
        self.residual = True
        self.node_update = E3NNConv(
                hidden_features, 
                hidden_features, 
                hidden_features,
                max_l=2,
                num_radial_channels=16,
                use_attention=False,
            )
        self.edge_update = EdgeGatedGraphConv(
            input_features,
            output_features
        )
    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        coord: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """
        Node, edge, and coord updates for EGNN layer.
        x: node input features
        y: edge input features
        coord: node coordinates
        z: edge pair input features
        """  
        g = g.local_var()
        lg = lg.local_var()

       # Edge-gated graph convolution update on crystal graph
        x, coord, m = self.node_update(g, x, coord, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, coord, y, z

class EGNNGraphConv(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
    ):
        """Initialize parameters for ALIEGNN update."""
        super().__init__()
        self.residual = True
        self.node_update = EGNNConv(
            input_features,
            hidden_features,
            output_features,
            hidden_features,
        )
        self.edge_update = EdgeGatedGraphConv(
            input_features,
            output_features
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        coord: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """
        Node, edge, and coord updates for EGNN layer.
        x: node input features
        y: edge input features
        coord: node coordinates
        z: edge pair input features
        """  
        g = g.local_var()
        lg = lg.local_var()

       # Edge-gated graph convolution update on crystal graph
        x, coord, m, _ = self.node_update(g, x, coord, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, coord, y, z

class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIEGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIEGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIEGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIEGNNConfig = ALIEGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.GatedE3NNGCN_layers = nn.ModuleList(
            [
                E3NNGraphConv(
                    config.hidden_features,
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.GatedEGCN_layers = nn.ModuleList(
            [
                EGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                    config.hidden_features,
                    config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )
        self.aliegnn_layers = nn.ModuleList(
            [
                EGNNGraphConv(
                    config.hidden_features,
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.egnn_gatv2_layers = nn.ModuleList(
            [
                EGNNGATv2Conv(
                    config.hidden_features,
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )
        self.egcn_layers = nn.ModuleList(
            [
                E3NNConv(
                config.hidden_features, 
                config.hidden_features, 
                config.hidden_features,
                max_l=2,
                num_radial_channels=16,
                use_attention=False,
                )
                for idx in range(config.gcn_layers)
            ]
        )
        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIEGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        vector = g.edata["r"]
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # initial coordinate features
        coord = g.ndata.pop("coords")

        # E3NN-enhanced EGNN layers
        for alie3nn_layer in self.GatedE3NNGCN_layers:
            x, coord, y, z = alie3nn_layer(g, lg, x, coord, y, z)

        # E3NN-enhanced GCN layers
        for egcn_layer in self.egcn_layers:
            x, coord, y = egcn_layer(g, x, coord, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)
