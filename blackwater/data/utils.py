"""Data utilities."""
from typing import Optional, List, Set, Dict, Union

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.providers import BackendV1
from torch_geometric.data import Data

available_gate_names = [
    # one qubit gates
    'id', 'u1', 'u2', 'u3', 'x', 'y',
    'z', 'h', 's', 'sdg', 't', 'tdg',
    'rx', 'ry', 'rz',
    # two qubit gates
    'cx', 'cy', 'cz', 'ch', 'crz',
    'cu1', 'cu3', 'swap', 'rzz',
    # three qubit gates
    'ccx', 'cswap'
]


def circuit_to_pyg_data(
    circuit: QuantumCircuit,
    gate_set: Optional[Set[str]] = None
) -> Data:
    """Convert circuit to Pytorch geometric data.
    Note: Homogenous conversion.

    Args:
        circuit: quantum circuit
        gate_set: list of instruction to use to encode data.
            if None, default instruction set will be used

    Returns:
        Pytorch geometric data
    """
    num_qubits = circuit.num_qubits
    gate_set: List[str] = gate_set or available_gate_names
    # add "other" gates
    gate_set += [
        "barrier",
        "measure",
        "delay"
    ]

    dag_circuit = circuit_to_dag(circuit)

    nodes = list(dag_circuit.nodes())
    edges = list(dag_circuit.edges())

    nodes_dict: Dict[DAGOpNode, List[float]] = {}

    for node_index, node in enumerate(nodes):
        if isinstance(node, (DAGInNode, DAGOutNode)):
            # TODO: use in and out nodes
            pass
        elif isinstance(node, DAGOpNode):
            # TODO: use node.cargs

            # get information on which qubits this gate is operating
            affected_qubits = [0.] * num_qubits
            for arg in node.qargs:
                affected_qubits[arg.index] = 1.0

            # encoding of gate name
            gate_encoding = [0.] * len(gate_set)
            gate_encoding[gate_set.index(node.op.name)] = 1.0

            # gate parameters
            node_params = [0., 0., 0.]
            for i, param in enumerate(node.op.params):
                node_params[i] = param

            feature_vector = gate_encoding + affected_qubits + node_params

            nodes_dict[node] = feature_vector

    nodes_indices = {node: idx for idx, node in enumerate(nodes_dict.keys())}

    edge_index = []
    edge_attr = []

    for edge in edges:
        source, dest, wire = edge

        if isinstance(source, DAGOpNode) and isinstance(dest, DAGOpNode):
            edge_index.append([nodes_indices[source], nodes_indices[dest]])
            edge_attr.append([0.])
        else:
            # TODO: handle in and out nodes
            pass

    return Data(
        x=torch.tensor(list(nodes_dict.values()), dtype=torch.float),
        edge_index=torch.tensor(np.transpose(edge_index), dtype=torch.long),
        edge_attr=torch.tensor(np.transpose(edge_attr), dtype=torch.float),
        circuit_depth=torch.tensor([[circuit.depth()]], dtype=torch.long)
    )


def circuit_backend_to_pyg_data(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: BackendV1
):
    """Convert circuit to Pytorch geometric data with backend information.

    Args:
        circuits: quantum circuits
        backend: backend to use to fetch information from

    Returns:
        Pytorch geometric data
    """
    pass
