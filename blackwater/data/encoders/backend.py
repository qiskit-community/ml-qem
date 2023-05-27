"""Backend encoders."""

from typing import Union

import torch
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.providers import BackendV1, BackendV2
from torch_geometric.data import Data

from blackwater.data.core import BackendEncoder
from blackwater.data.encoders.graph_utils import backend_to_json_graph

N_QUBIT_PROPERTIES = 2
ALL_INSTRUCTIONS = list(get_standard_gate_name_mapping().keys())


# pylint: disable=no-member
class DefaultPyGBackendEncoder(BackendEncoder):
    """Default pytorch geometric backend encoder.

    Turns backend into pyg data.
    """

    def encode(self, backend: Union[BackendV1, BackendV2], **kwargs):  # type: ignore
        backend_graph = backend_to_json_graph(backend)
        backend_nodes = torch.tensor(backend_graph.nodes, dtype=torch.float)
        backend_edges = torch.transpose(
            torch.tensor(backend_graph.edges, dtype=torch.float), 0, 1
        )
        backend_edge_features = torch.tensor(
            backend_graph.edge_features, dtype=torch.float
        )
        return Data(
            x=backend_nodes, edge_index=backend_edges, edge_attr=backend_edge_features
        )
