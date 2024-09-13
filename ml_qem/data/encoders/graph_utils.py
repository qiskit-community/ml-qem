"""Graph encoders."""

from dataclasses import dataclass, asdict
from typing import Union, List, Dict, Optional, Tuple

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGNode, DAGOpNode
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1, BackendV2, QubitProperties
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.transpiler import Target
from torch_geometric.data import Data

from ml_qem.data.core import DataEncoder, MLQEMData, NodeEncoder
from ml_qem.data.encoders.utils import OperatorData, encode_operator
from ml_qem.exception import MLQEMException

N_QUBIT_PROPERTIES = 2
ALL_INSTRUCTIONS = list(get_standard_gate_name_mapping().keys())


# pylint: disable=no-member, arguments-differ
class DefaultNodeEncoder(NodeEncoder):
    """DefaultNodeEncoder."""

    def __init__(self, available_instructions: Optional[List[str]] = None):
        """Default node encoder.

        Only encodes type of gate and parameters.

        Args:
            available_instructions: list of available instructions for encoding.
                Default all Qiskit instructions.
        """
        available_instructions = available_instructions or ALL_INSTRUCTIONS

        default_instructions = ["barrier", "measure", "reset"]
        for inst in default_instructions:
            if inst not in available_instructions:
                available_instructions.append(inst)
        inst_size = len(available_instructions)

        self.encoding_map = {
            inst: [0 if i != idx else 1 for i in range(inst_size)]
            for idx, inst in enumerate(available_instructions)
        }

    def encode(self, node: DAGNode, **kwargs) -> List[float]:  # type: ignore
        if isinstance(node, DAGOpNode):
            params_encoding = [0.0, 0.0, 0.0]
            for i, param in enumerate(node.op.params):
                if isinstance(param, (float, int)):
                    params_encoding[i] = float(param)
                elif param.is_real():
                    params_encoding[i] = float(param._symbol_expr)

            name_encoding = self.encoding_map[node.op.name]
            result = name_encoding + params_encoding
        else:
            raise NotImplementedError(
                f"Node type {type(node)} is not supported by encoder yet."
            )

        return result


@dataclass
class BackendProperties:
    """BackendProperties."""

    qubit_properties_map: Dict[int, List[float]]
    gate_properties_map: Optional[Dict[str, List[float]]] = None


def extract_properties_from_backend(
    backend: Union[BackendV1, BackendV2]
) -> BackendProperties:
    """Returns backend properties.

    Args:
        backend: backend

    Returns:
        BackendProperties
    """
    if isinstance(backend, BackendV2):
        target: Target = backend.target

        # qubit properties
        qubit_properties: List[QubitProperties] = target.qubit_properties
        qubit_properties_map = {}
        for idx, qprops in enumerate(qubit_properties):
            if (
                isinstance(qprops.t1, (float, int))
                and isinstance(qprops.t2, (float, int))
                and isinstance(qprops.frequency, (float, int))
            ):
                qubit_properties_map[idx] = [qprops.t1, qprops.t2]
            else:
                qubit_properties_map[idx] = [0.0] * N_QUBIT_PROPERTIES

        return BackendProperties(qubit_properties_map)

    else:
        raise MLQEMException(f"Backend of type [{type(backend)}] is not supported yet.")


class BackendNodeEncoder(NodeEncoder):
    """BackendNodeEncoder."""

    def __init__(self, backend: BackendV2):
        """Circuit node encoder based on backend properties.

        Args:
            backend: backend
        """
        available_instructions = backend.operation_names
        default_instructions = ["barrier", "measure", "reset"]
        for inst in default_instructions:
            if inst not in available_instructions:
                available_instructions.append(inst)
        inst_size = len(available_instructions)

        self.encoding_map = {
            inst: [0 if i != idx else 1 for i in range(inst_size)]
            for idx, inst in enumerate(available_instructions)
        }
        self.backend = backend
        self.num_qubits = backend.num_qubits
        self.properties: BackendProperties = extract_properties_from_backend(backend)

    def encode(self, node: DAGNode, **kwargs) -> List[float]:  # type: ignore
        if isinstance(node, DAGOpNode):
            params_encoding = [0.0, 0.0, 0.0]
            for i, param in enumerate(node.op.params):
                if isinstance(param, (float, int)):
                    params_encoding[i] = float(param)
                elif param.is_real():
                    params_encoding[i] = float(param._symbol_expr)

            if node.op.name not in self.encoding_map:
                raise MLQEMException(
                    f"Instruction [{node.op.name}] is not available"
                    f" for backend [{self.backend.name}]. "
                    f"Maybe you forgot to transpile circuit for this backend?"
                )
            name_encoding = self.encoding_map[node.op.name]

            qubit_properties_encoding = [0.0] * (self.num_qubits * N_QUBIT_PROPERTIES)
            for qubit in node.qargs:
                if isinstance(qubit, Qubit):
                    qubit_index = qubit.index
                    for i, value in enumerate(
                        self.properties.qubit_properties_map.get(
                            qubit_index, [0.0] * N_QUBIT_PROPERTIES
                        )
                    ):
                        qubit_properties_encoding[
                            qubit_index * N_QUBIT_PROPERTIES + i
                        ] = value

            result = name_encoding + params_encoding + qubit_properties_encoding
        else:
            raise NotImplementedError(
                f"Node type {type(node)} is not supported by BackendNodeEncoder yet."
            )
        return result


@dataclass
class GraphData:
    """GraphData."""

    nodes: List[List[float]]
    edges: List[List[float]]
    edge_features: List[List[float]]


def circuit_to_json_graph(
    circuit: QuantumCircuit,
    available_instructions: Optional[List[str]] = None,
    node_types_to_encode: Optional[List[type]] = None,
    node_encoder: Optional[NodeEncoder] = None,
) -> GraphData:
    """Encodes circuit to json graph data

    Args:
        circuit: circuit
        available_instructions: list of available instructions. Default all Qiskit instructions.
        node_types_to_encode: list of types of nodes to encode
        node_encoder: node encoder. Default: DefaultNodeEncoder

    Returns:
        dictionary with encoded circuit as graph
    """
    node_encoder = node_encoder or DefaultNodeEncoder(
        available_instructions=available_instructions
    )
    node_types_to_encode: List[type] = node_types_to_encode or [DAGOpNode]  # type: ignore[no-redef]

    dag_circuit = circuit_to_dag(circuit)

    nodes_map: Dict[str, List[float]] = {
        str(node): node_encoder.encode(node)
        for node in list(dag_circuit.nodes())
        if isinstance(node, tuple(node_types_to_encode))
    }
    node_indexer = {node: idx for idx, node in enumerate(nodes_map.keys())}
    edges = [
        [node_indexer[str(src)], node_indexer[str(dst)]]
        for (src, dst, wire) in list(dag_circuit.edges())
        if isinstance(src, tuple(node_types_to_encode))
        and isinstance(dst, tuple(node_types_to_encode))
    ]

    edge_features = [[0.0] for _ in range(len(edges))]
    nodes = list(nodes_map.values())
    return GraphData(nodes=nodes, edges=edges, edge_features=edge_features)  # type: ignore[arg-type]


def backend_to_json_graph(backend: Union[BackendV1, BackendV2]) -> GraphData:
    """Encodes backend to json graph data

    Args:
        backend: backend

    Returns:
        dictionary with encoded data as graph
    """
    if isinstance(backend, BackendV1):
        raise MLQEMException("BackendV1 is not supported yet.")

    target: Target = backend.target

    qubit_properties: List[QubitProperties] = target.qubit_properties
    nodes = [
        [properties.t1, properties.t2, properties.frequency]
        for properties in qubit_properties
    ]

    edges = [list(qubits) for inst, qubits in target.instructions if len(qubits) == 2]

    edge_features = [[0.0] for _ in range(len(edges))]

    return GraphData(nodes=nodes, edges=edges, edge_features=edge_features)


@dataclass
class PygData(MLQEMData):
    """PygData."""

    @classmethod
    def deserialize(cls, data: dict):
        raise NotImplementedError

    def serialize(self) -> dict:
        return asdict(self)

    def to_pyg(self) -> Data:
        """Converts to pytorch_geometric.Data object."""
        raise NotImplementedError


@dataclass
class ExpValData(PygData):
    """ExpValData."""

    circuit: GraphData
    circuit_depth: int
    expectation_values: List[float]
    observable: Optional[OperatorData] = None
    backend: Optional[GraphData] = None

    @classmethod
    def build(
        cls,
        circuit: QuantumCircuit,
        expectation_values: List[float],
        observable: Optional[BaseOperator] = None,
        backend: Optional[Union[BackendV1, BackendV2]] = None,
    ):
        """Constructs ExpValData from Qiskit classes.

        Args:
            circuit: circuit
            expectation_values: list of expectation values
            observable: observable
            backend: backend

        Returns:
            ExpValData object
        """
        node_encoder = (
            DefaultNodeEncoder() if backend is None else BackendNodeEncoder(backend)
        )
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
            backend=encoded_backend,
        )

    def to_pyg(self):
        optional_data = {}

        circuit_nodes = torch.tensor(self.circuit.nodes, dtype=torch.float)
        circuit_edges = torch.transpose(
            torch.tensor(self.circuit.edges, dtype=torch.long), 0, 1
        )
        circuit_edge_features = torch.tensor(
            self.circuit.edge_features, dtype=torch.float
        )
        circuit_depth = torch.tensor([[self.circuit_depth]], dtype=torch.float)

        expectation_values = torch.tensor([self.expectation_values], dtype=torch.float)

        if self.observable is not None:
            optional_data["observable"] = torch.tensor(
                [self.observable.operator], dtype=torch.float
            )

        if self.backend is not None:
            optional_data["backend_nodes"] = torch.tensor(
                self.backend.nodes, dtype=torch.float
            )
            optional_data["backend_edges"] = torch.transpose(
                torch.tensor(self.backend.edges, dtype=torch.float), 0, 1
            )
            optional_data["backend_edge_features"] = torch.tensor(
                self.backend.edge_features, dtype=torch.float
            )

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
                entry_data[key] = OperatorData(**value)  # type: ignore[assignment]
            else:
                entry_data[key] = value
        return cls(**entry_data)  # type: ignore[arg-type]


class DefaultPyGEstimatorEncoder(DataEncoder):
    """Default encoder for pyg data.
    Converts circuit data into torch_geometric.Data"""

    def encode(  # type: ignore
        self,
        circuit: QuantumCircuit,
        operator: PauliSumOp,
        exp_val: float,
        backend: BackendV2,
        **kwargs,
    ) -> Tuple[Data, float]:
        data = ExpValData.build(
            circuit=circuit,
            expectation_values=[exp_val],
            observable=operator,
            backend=backend,
        ).to_pyg()
        return data, exp_val
