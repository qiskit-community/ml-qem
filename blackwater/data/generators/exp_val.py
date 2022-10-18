import random
from dataclasses import dataclass
from typing import List, Iterator, Dict, Any

import torch
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers import BackendV1
from torch_geometric.data import Data

from blackwater.data.utils import generate_random_pauli_sum_op, create_estimator_meas_data, circuit_to_graph_data_json, \
    get_backend_properties_v1, encode_pauli_sum_op


@dataclass
class CircuitGraphData:
    nodes: List[List[float]]
    edge_index: List[List[int]]
    edge_attr: List[List[float]]


@dataclass
class ExpValueEntry:
    circuit_graph: Dict[str, Any]
    observable: List[List[float]]
    ideal_exp_value: float
    noisy_exp_value: float
    circuit_depth: int = 0

    def to_dict(self):
        return {
            "circuit_graph": self.circuit_graph,
            "observable": self.observable,
            "ideal_exp_value": self.ideal_exp_value,
            "noisy_exp_value": self.noisy_exp_value,
            "circuit_depth": self.circuit_depth
        }

    @classmethod
    def from_json(cls, dictionary: Dict[str, Any]):
        return ExpValueEntry(**dictionary)

    def to_pyg_data(self):
        key = 'DAGOpNode_wire_DAGOpNode'
        g_data = self.circuit_graph

        x = torch.tensor(g_data["nodes"]["DAGOpNode"], dtype=torch.float)
        edge_index = torch.tensor(g_data["edges"][key]["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(g_data["edges"][key]["edge_attr"], dtype=torch.float)
        y = torch.tensor([[self.ideal_exp_value]], dtype=torch.float)
        noisy = torch.tensor([[self.noisy_exp_value]], dtype=torch.float)
        observable = torch.tensor([self.observable], dtype=torch.float)
        circuit_depth = torch.tensor([[self.circuit_depth]], dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            noisy=noisy,
            observable=observable,
            circuit_depth=circuit_depth
        )


def exp_value_generator(
    backend: BackendV1,
    n_qubits: int,
    circuit_depth: int,
    pauli_terms: int,
    pauli_coeff: float = 1.0,
    max_entries: int = 1000
) -> Iterator[ExpValueEntry]:
    properties = get_backend_properties_v1(backend)

    for _ in range(max_entries):
        circuit = transpile(random_circuit(n_qubits, random.randint(1, circuit_depth)),
                            backend,
                            optimization_level=0)
        graph_data = circuit_to_graph_data_json(circuit=circuit, properties=properties)
        observable = generate_random_pauli_sum_op(n_qubits, pauli_terms, pauli_coeff)

        ideal_exp_val, noisy_exp_val = create_estimator_meas_data(
            backend=backend, circuit=circuit, observable=observable
        )

        yield ExpValueEntry(
            circuit_graph=graph_data,
            observable=encode_pauli_sum_op(observable),
            ideal_exp_value=ideal_exp_val,
            noisy_exp_value=noisy_exp_val
        )
