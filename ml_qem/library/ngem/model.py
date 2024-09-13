"""Models."""

import torch
from torch_geometric.nn import (
    TransformerConv,
    global_mean_pool,
    Linear,
    ASAPooling,
)


# pylint: disable=no-member
class SimpleExpectationValueModel(torch.nn.Module):
    """SimpleExpectationValueModel."""

    def __init__(
        self, num_node_features: int, num_obs_features: int, hidden_channels: int
    ):
        """Simple model for mitigation of exp values.

        Args:
            num_node_features: num of node features
            num_obs_features: num of obs features
            hidden_channels: number of channels in hidden layers
        """
        super().__init__()

        self.transformer1 = TransformerConv(
            num_node_features, hidden_channels, heads=3, dropout=0.1
        )
        self.pooling1 = ASAPooling(hidden_channels * 3, 0.5)

        self.transformer2 = TransformerConv(
            hidden_channels * 3, 1, heads=2, dropout=0.1
        )
        self.pooling2 = ASAPooling(2, 0.5)

        self.obs_seq = torch.nn.Sequential(
            Linear(num_obs_features, hidden_channels),
            torch.nn.Dropout(0.2),
            Linear(hidden_channels, 1),
        )

        self.body_seq = torch.nn.Sequential(
            Linear(5, hidden_channels),
            torch.nn.Dropout(0.2),
            Linear(hidden_channels, 1),
        )

    def forward(
        self, circuit_nodes, edges, noisy_exp_value, observable, circuit_depth, batch
    ):
        """Forward pass."""
        graph = self.transformer1(circuit_nodes, edges)
        graph, edges, _, batch, _ = self.pooling1(graph, edges, batch=batch)

        graph = self.transformer2(graph, edges)
        graph, edges, _, batch, _ = self.pooling2(graph, edges, batch=batch)

        graph = global_mean_pool(graph, batch)

        obs = self.obs_seq(observable)
        obs = torch.mean(obs.flatten(1, 2), 1, True)

        merge = torch.cat(
            (
                graph,
                noisy_exp_value,
                obs,
                circuit_depth,
            ),
            dim=1,
        )
        res = self.body_seq(merge)

        return res
