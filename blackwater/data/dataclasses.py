from dataclasses import dataclass, asdict
from typing import Optional, List, Union

import torch
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, BackendV1
from qiskit.quantum_info.operators.base_operator import BaseOperator
from torch_geometric.data import Data

from .encoders.graph import GraphData, backend_to_json_graph, circuit_to_json_graph
from .encoders.node_encoder import DefaultNodeEncoder, BackendNodeEncoder
from .encoders.operator import OperatorData
from .encoders.operator import encode_operator


@dataclass
class BlackwaterData:
    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict):
        raise NotImplementedError


@dataclass
class PygData(BlackwaterData):
    @classmethod
    def deserialize(cls, data: dict):
        raise NotImplementedError

    def serialize(self) -> dict:
        return asdict(self)

    def to_pyg(self) -> Data:
        raise NotImplementedError


@dataclass
class ExpValData(PygData):
    circuit: GraphData
    circuit_depth: int
    expectation_values: List[float]
    observable: Optional[OperatorData] = None
    backend: Optional[GraphData] = None

    @classmethod
    def build(cls,
              circuit: QuantumCircuit,
              expectation_values: List[float],
              observable: Optional[BaseOperator] = None,
              backend: Optional[Union[BackendV1, BackendV2]] = None):
        node_encoder = DefaultNodeEncoder() if backend is None else BackendNodeEncoder(backend)
        encoded_observable = None
        if observable is not None:
            encoded_observable = encode_operator(observable)

        encoded_backend = None
        if backend is not None:
            encoded_backend = backend_to_json_graph(backend)

        return cls(
            circuit=circuit_to_json_graph(circuit=circuit, node_encoder=node_encoder),
            circuit_depth=circuit.depth(),
            expectation_values=expectation_values,
            observable=encoded_observable,
            backend=encoded_backend
        )

    def to_pyg(self):
        optional_data = {}

        circuit_nodes = torch.tensor(self.circuit.nodes, dtype=torch.float)
        circuit_edges = torch.transpose(torch.tensor(self.circuit.edges, dtype=torch.long), 0, 1)
        circuit_edge_features = torch.tensor(self.circuit.edge_features, dtype=torch.float)
        circuit_depth = torch.tensor([[self.circuit_depth]], dtype=torch.float)

        expectation_values = torch.tensor([self.expectation_values], dtype=torch.float)

        if self.observable is not None:
            optional_data["observable"] = torch.tensor([self.observable.operator], dtype=torch.float)

        if self.backend is not None:
            optional_data["backend_nodes"] = torch.tensor(self.backend.nodes, dtype=torch.float)
            optional_data["backend_edges"] = torch.transpose(torch.tensor(self.backend.edges, dtype=torch.float), 0, 1)
            optional_data["backend_edge_features"] = torch.tensor(self.backend.edge_features, dtype=torch.float)

        return Data(
            x=circuit_nodes,
            edge_index=circuit_edges,
            edge_attr=circuit_edge_features,
            y=expectation_values,
            circuit_depth=circuit_depth,
            **optional_data,
        )

    @classmethod
    def deserialize(cls, data: dict):
        entry_data = {}
        for key, value in data.items():
            if key in ["circuit", "backend"]:
                entry_data[key] = GraphData(**value)
            elif key in ["observable"]:
                entry_data[key] = OperatorData(**value)
            else:
                entry_data[key] = value
        return cls(**entry_data)
