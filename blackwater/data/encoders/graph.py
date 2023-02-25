"""Graph encoders."""
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers import BackendV2, BackendV1, QubitProperties
from qiskit.transpiler import Target

from .node_encoder import NodeEncoder, DefaultNodeEncoder
from ...exception import BlackwaterException


@dataclass
class GraphData:
    nodes: List[List[float]]
    edges: List[List[float]]
    edge_features: List[List[float]]


def circuit_to_json_graph(
        circuit: QuantumCircuit,
        available_instructions: Optional[List[str]] = None,
        node_types_to_encode: Optional[List[type]] = None,
        node_encoder: Optional[NodeEncoder] = None
) -> GraphData:
    node_encoder = node_encoder or DefaultNodeEncoder(available_instructions=available_instructions)
    node_types_to_encode: List[type] = node_types_to_encode or [DAGOpNode]

    dag_circuit = circuit_to_dag(circuit)

    nodes_map: Dict[str, List[float]] = {
        str(node): node_encoder.encode(node) for node in
        list(dag_circuit.nodes())
        if isinstance(node, tuple(node_types_to_encode))
    }
    node_indexer = {
        node: idx for idx, node in
        enumerate(nodes_map.keys())
    }
    edges = [
        [node_indexer[str(src)], node_indexer[str(dst)]]
        for (src, dst, wire) in list(dag_circuit.edges())
        if isinstance(src, tuple(node_types_to_encode))
           and isinstance(dst, tuple(node_types_to_encode))
    ]

    edge_features = [
        [0.] for _ in range(len(edges))
    ]

    return GraphData(
        nodes=list(nodes_map.values()),
        edges=edges,
        edge_features=edge_features
    )


def backend_to_json_graph(backend: Union[BackendV1, BackendV2]) -> GraphData:
    if isinstance(backend, BackendV1):
        raise BlackwaterException("BackendV1 is not supported yet.")

    target: Target = backend.target

    qubit_properties: List[QubitProperties] = target.qubit_properties
    nodes = [
        [properties.t1, properties.t2, properties.frequency]
        for properties in qubit_properties
    ]

    edges = [
        list(qubits)
        for inst, qubits in target.instructions
        if len(qubits) == 2
    ]

    edge_features = [
        [0.] for _ in range(len(edges))
    ]

    return GraphData(
        nodes=nodes,
        edges=edges,
        edge_features=edge_features
    )
