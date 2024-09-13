"""Circuit encoders."""

from typing import Optional

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from torch_geometric.data import Data

from ml_qem.data.core import CircuitEncoder
from ml_qem.data.encoders.graph_utils import (
    DefaultNodeEncoder,
    circuit_to_json_graph,
    BackendNodeEncoder,
)


# pylint: disable=no-member
class DefaultCircuitEncoder(CircuitEncoder):
    """Default circuit encoder to transform circuit into numpy array for training

    Returns:
        numpy array where:
        - first element - depth of circuit
        - second element - 2q depth of circuit
        - 3rd - number of 1q gates
        - 4th - number of 2q gates
        - 5th - num qubits
    """

    def encode(self, circuit: QuantumCircuit, **kwargs) -> np.ndarray:  # type: ignore
        """Encodes circuit.

        Args:
            circuit: circuit to encoder
            **kwargs: other arguments

        Returns:
            numpy array
        """
        depth = circuit.depth()
        two_qubit_depth = circuit.depth(lambda x: x[0].num_qubits == 2)

        num_one_q_gates = 0
        num_two_q_gates = 0
        for instr in circuit._data:
            num_qubits = len(instr.qubits)
            if num_qubits == 1:
                num_one_q_gates += 1
            if num_qubits == 2:
                num_two_q_gates += 1

        return np.array(
            [
                depth,
                two_qubit_depth,
                num_one_q_gates,
                num_two_q_gates,
                circuit.num_qubits,
            ]
        )


class DefaultPyGCircuitEncoder(CircuitEncoder):
    """Default pytorch geometric circuit encoder.

    Turns circuit into pyg data.
    """

    def __init__(self, backend: Optional[BackendV2]):
        """Constructor.

        Args:
            backend: optional backend. Will be used for node data encoding.
        """
        self.backend = backend

    def encode(self, circuit: QuantumCircuit, **kwargs):  # type: ignore
        node_encoder = (
            DefaultNodeEncoder()
            if self.backend is None
            else BackendNodeEncoder(self.backend)
        )
        circuit_graph = circuit_to_json_graph(circuit, node_encoder=node_encoder)
        circuit_nodes = torch.tensor(circuit_graph.nodes, dtype=torch.float)
        circuit_edges = torch.transpose(
            torch.tensor(circuit_graph.edges, dtype=torch.long), 0, 1
        )
        circuit_edge_features = torch.tensor(
            circuit_graph.edge_features, dtype=torch.float
        )
        return Data(
            x=circuit_nodes, edge_index=circuit_edges, edge_attr=circuit_edge_features
        )
