"""Torch Module for E(n) Equivariant Graph Convolutional Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class EGNNConv(nn.Module):
    """Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::

        m_{ij}=\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})

        x_i^{l+1} = x_i^l + C\sum_{j\in\mathcal{N}(i)}(x_i^l-x_j^l)\phi_x(m_{ij})

        m_i = \sum_{j\in\mathcal{N}(i)} m_{ij}

        h_i^{l+1} = \phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\phi_e`, :math:`\phi_h`, and
    :math:`\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\phi_e, \phi_x, \phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    Example
    ----------
    """

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNConv, self).__init__()
        self.num_heads = 1
        self.atten_gate = nn.Linear(hidden_size, self.num_heads)
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
        )

        self.dst_update = nn.Linear(hidden_size, hidden_size)

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            act_fn,
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["radial"],
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data['radial']], dim=-1
            )

        msg_h = self.edge_mlp(f)
        e = self.atten_gate(msg_h)
        # apply attention gate

        return {"atten_e": e, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        """
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        with graph.local_scope():
            # node feature
            graph.ndata["h"] = node_feat
            # coordinate feature
            graph.ndata["x"] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            #graph.edata["radial"] = (
            #    graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1)
            #)
            graph.edata["radial"] = torch.norm(graph.edata["x_diff"], dim=1, keepdim=True)
            
            a = graph.edata["x_diff"]
            # normalize coordinate difference
            eps = 1e-6  # 增大epsilon值
            graph.edata["x_diff"] = graph.edata["x_diff"] / (graph.edata["radial"] + eps)
            graph.apply_edges(self.message)

            # apply attention gate
            graph.edata["alpha"] = dgl.ops.edge_softmax(graph, graph.edata["atten_e"])
            graph.update_all(
                message_func=lambda edges: {"m_x": edges.data["alpha"].unsqueeze(-1) * edges.data["x_diff"].unsqueeze(1)},
                reduce_func=fn.sum("m_x", "msg_x")
            )

            graph.ndata["x_neigh"] = torch.mean(graph.ndata["msg_x"], dim=1)
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))
            # graph.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "h_neigh"))
            # graph.ndata["h"] = graph.ndata["sum_h_sigma"] / (
            #     graph.ndata["h_neigh"] + 1e-6
            # )

            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh
            # edge feature update
            if self.edge_feat_size > 0:
                e = edge_feat + self.edge_gate(graph.edata["msg_h"])
                return h, x, e, a
            else:
                return h, x, a
