"""Data utilities."""
from typing import Optional, List, Dict, Union, Any

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.providers import BackendV1
from qiskit.quantum_info import random_pauli_list, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
from torch_geometric.data import Data

# pylint: disable=no-member
available_gate_names = [
    # one qubit gates
    "id",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    # two qubit gates
    "cx",
    "cy",
    "cz",
    "ch",
    "crz",
    "cu1",
    "cu3",
    "swap",
    "rzz",
    # three qubit gates
    "ccx",
    "cswap",
]


def circuit_to_pyg_data(
    circuit: QuantumCircuit, gate_set: Optional[List[str]] = None
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
    gate_set = gate_set or available_gate_names
    # add "other" gates
    gate_set += ["barrier", "measure", "delay"]

    dag_circuit = circuit_to_dag(circuit)

    nodes = list(dag_circuit.nodes())
    edges = list(dag_circuit.edges())

    nodes_dict: Dict[DAGOpNode, List[float]] = {}

    for _, node in enumerate(nodes):
        if isinstance(node, (DAGInNode, DAGOutNode)):
            # TODO: use in and out nodes
            pass
        elif isinstance(node, DAGOpNode):
            # TODO: use node.cargs

            # get information on which qubits this gate is operating
            affected_qubits = [0.0] * num_qubits
            for arg in node.qargs:
                affected_qubits[arg.index] = 1.0

            # encoding of gate name
            gate_encoding = [0.0] * len(gate_set)
            gate_encoding[gate_set.index(node.op.name)] = 1.0

            # gate parameters
            node_params = [0.0, 0.0, 0.0]
            for i, param in enumerate(node.op.params):
                node_params[i] = param

            feature_vector = gate_encoding + affected_qubits + node_params

            nodes_dict[node] = feature_vector

    nodes_indices = {node: idx for idx, node in enumerate(nodes_dict.keys())}

    edge_index = []
    edge_attr = []

    for edge in edges:
        source, dest, _ = edge

        if isinstance(source, DAGOpNode) and isinstance(dest, DAGOpNode):
            edge_index.append([nodes_indices[source], nodes_indices[dest]])
            edge_attr.append([0.0])
        else:
            # TODO: handle in and out nodes
            pass

    return Data(
        x=torch.tensor(list(nodes_dict.values()), dtype=torch.float),
        edge_index=torch.tensor(np.transpose(edge_index), dtype=torch.long),
        edge_attr=torch.tensor(np.transpose(edge_attr), dtype=torch.float),
        circuit_depth=torch.tensor([[circuit.depth()]], dtype=torch.long),
    )


def gate_to_index(gate: Any):
    """Converts gate to key:
        f(gate(cx, [0, 1])) -> 'cx_0_1'

    Args:
        gate: gate

    Returns:
        key
    """
    return f"{gate.gate}_{'_'.join([str(i) for i in gate.qubits])}"


def get_backend_properties_v1(backend: BackendV1):
    """Get properties from BackendV1

    Args:
        backend: backend

    Returns:
        json with backend information
    """
    props = backend.properties()

    def get_parameters(gate):
        return {
            **{"gate_error": 0.0, "gate_length": 0.0},
            **{param.name: param.value for param in gate.parameters},
        }

    return {
        "name": backend.name(),
        "gates_set": list({g.gate for g in props.gates}),
        "num_qubits": len(props.qubits),
        "qubits_props": {
            index: {
                "index": index,
                "t1": props.qubit_property(index).get("T1", (0, 0))[0],
                "t2": props.qubit_property(index).get("T2", (0, 0))[0],
                "readout_error": props.qubit_property(index).get(
                    "readout_error", (0, 0)
                )[0],
            }
            for index in range(len(props.qubits))
        },
        "gate_props": {
            gate_to_index(gate): {"index": gate_to_index(gate), **get_parameters(gate)}
            for gate in props.gates
        },
    }


def counts_to_feature_vector(counts: dict, num_qubits: int) -> List[float]:
    """Convert counts to feature vector.

    Args:
        counts: counts
        num_qubits: number of qubits

    Returns:
        list of floats
    """
    count_format = "{:0" + str(num_qubits) + "b}"
    all_possible_measurements = {
        count_format.format(i): 0 for i in range(2**num_qubits)
    }

    shots = sum(counts.values())
    all_counts = {**all_possible_measurements, **counts}
    return list(float(v) / shots for v in all_counts.values())


def circuit_to_graph_data_json(
    circuit: QuantumCircuit,
    properties: dict,
    use_gate_features: bool = False,
    use_qubit_features: bool = False,
):
    """Converts circuit to json (dict) for PyG data.

    Args:
        circuit: quantum circuit
        properties: call get_backend_properties_v1 for backend
        use_gate_features: use gate features in data graph
        use_qubit_features: use qubit features in data graph
    """

    # feature map for gate types
    additional_gate_types = [
        "barrier",
        "measure",
        # "delay"
    ]
    gate_type_feature_map = {
        g_name: index
        for index, g_name in enumerate(properties["gates_set"] + additional_gate_types)
    }

    # convert to dag
    dag_circuit = circuit_to_dag(circuit)

    nodes = list(dag_circuit.nodes())
    edges = list(dag_circuit.edges())

    # get node data
    nodes_dict: Dict[str, Dict[str, Any]] = {
        "DAGOpNode": {},
        "DAGInNode": {},
        "DAGOutNode": {},
    }

    circuit_n_qubits = circuit.num_qubits

    for node in nodes:
        if isinstance(node, DAGOpNode):

            # qubit features
            qubit_properties = {i: {} for i in range(3)}  # as 3 is max number of operable gate size
            if node.name != 'barrier' and len(node.qargs) > 3:
                raise Exception("Non barrier gate that has more than 3 qubits."
                                "Those tyoe of gates are not supported yet.")

            if node.name != 'barrier':  # barriers are more than 3 qubits
                for i, qubit in enumerate(node.qargs):
                    qubit_properties[i] = properties["qubits_props"][qubit.index]

            t1_vector = [v.get("t1", 0.0) for v in qubit_properties.values()]
            t2_vector = [v.get("t2", 0.0) for v in qubit_properties.values()]
            readout_error_vector = [
                v.get("readout_error", 0.0) for v in qubit_properties.values()
            ]

            qubit_feature_vector = t1_vector + t2_vector + readout_error_vector

            # gate features
            index = len(list(nodes_dict["DAGOpNode"].keys()))

            instruction_key = (
                f"{node.op.name}_"
                f"{'_'.join([str(args.index) for args in list(node.qargs)])}"
            )

            gate_props = properties["gate_props"].get(instruction_key, {})
            gate_props = {**{"gate_error": 0.0, "gate_length": 0.0}, **gate_props}
            if "index" in gate_props:
                del gate_props["index"]

            # one hot encoding of gate type
            gate_type_feature = [0.0 for _ in range(len(gate_type_feature_map))]
            gate_type_feature[gate_type_feature_map[node.op.name]] = 1.0

            # gate parameter values feature vector
            gate_params_feature_vector = [
                0.0,
                0.0,
                0.0,
            ]  # 3 is max number of parameter in any instruction
            for idx, p in enumerate(node.op.params):
                if isinstance(p, (float, int)):
                    gate_params_feature_vector[idx] = float(p)
                elif p.is_real():
                    gate_params_feature_vector[idx] = float(p._symbol_expr)

            # gate properties
            gate_props_feature_vector = [
                gate_props["gate_error"],
                gate_props["gate_length"],
            ]

            feature_vector = gate_params_feature_vector + gate_type_feature
            if use_qubit_features:
                feature_vector += qubit_feature_vector
            if use_gate_features:
                feature_vector += gate_props_feature_vector

            nodes_dict["DAGOpNode"][node] = {
                "index": index,
                "type": "DAGOpNode",
                "name": node.op.name,
                "num_qubits": node.op.num_qubits,
                "num_clbits": node.op.num_clbits,
                "params": [],
                "feature_vector": feature_vector,
                **gate_props,
            }

        elif isinstance(node, (DAGInNode, DAGOutNode)):
            node_type_key = str(type(node)).split("'")[1].split(".")[-1]
            index = len(list(nodes_dict[node_type_key].keys()))
            nodes_dict[node_type_key][node] = {
                "index": index,
                "type": node_type_key,
                "register": node.wire.register.name,
                "bit": node.wire.index,
                "feature_vector": [0, 0],
            }

    # get edge data
    edge_dict = {}

    for edge in edges:
        source, dest, wire = edge
        source_type = str(type(source)).split("'")[1].split(".")[-1]
        dest_type = str(type(dest)).split("'")[1].split(".")[-1]

        source = nodes_dict[source_type][source]
        dest = nodes_dict[dest_type][dest]

        if isinstance(wire, Qubit):
            edge_attrs = properties["qubits_props"][wire.index]
            key = (source_type, "wire", dest_type)

            if key not in edge_dict:
                edge_dict[key] = {
                    "edge_index": [[source["index"], dest["index"]]],
                    "edge_attr": [
                        [
                            edge_attrs["t1"],
                            edge_attrs["t2"],
                            edge_attrs["readout_error"],
                        ]
                    ],
                }
            else:
                edge_dict[key]["edge_index"].append([source["index"], dest["index"]])
                edge_dict[key]["edge_attr"].append(
                    [edge_attrs["t1"], edge_attrs["t2"], edge_attrs["readout_error"]]
                )

    # form data
    data: Dict[str, Dict[str, Any]] = {"nodes": {}, "edges": {}}

    data["nodes"]["DAGOpNode"] = [
        node["feature_vector"] for node in nodes_dict["DAGOpNode"].values()
    ]
    data["nodes"]["DAGInNode"] = [
        node["feature_vector"] for node in nodes_dict["DAGInNode"].values()
    ]
    data["nodes"]["DAGOutNode"] = [
        node["feature_vector"] for node in nodes_dict["DAGOutNode"].values()
    ]

    for key, d in edge_dict.items():
        edge_index = np.array(d["edge_index"]).T.tolist()
        edge_attr = d["edge_attr"]

        data["edges"]["_".join(list(key))] = {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

    # # get measurements
    # sim_ideal = AerSimulator()
    # sim_noisy = AerSimulator.from_backend(backend)
    #
    # result_ideal = sim_ideal.run(circuit).result().get_counts()
    # result_noisy = sim_noisy.run(circuit).result().get_counts()
    #
    # data["y"] = {
    #     "ideal": counts_to_feature_vector(result_ideal, properties["num_qubits"]),
    #     "nosiy": counts_to_feature_vector(result_noisy, properties["num_qubits"])
    # }

    return data


def create_counts_meas_data(
    backend: BackendV1, circuit: QuantumCircuit, properties: Dict[str, Any]
):
    """Creates counts measurement for circuit

    Args:
        backend: backend
        circuit: circuit
        properties: backend properties

    Returns:
        dict of ideal and noisy measurements
    """
    # get measurements
    sim_ideal = AerSimulator()
    sim_noisy = AerSimulator.from_backend(backend)

    result_ideal = sim_ideal.run(circuit).result().get_counts()
    result_noisy = sim_noisy.run(circuit).result().get_counts()

    return {
        "ideal": counts_to_feature_vector(result_ideal, properties["num_qubits"]),
        "nosiy": counts_to_feature_vector(result_noisy, properties["num_qubits"]),
    }


def create_estimator_meas_data(
    backend: BackendV1, circuit: QuantumCircuit, observable: PauliSumOp
):
    """Runs Aer estimator with noisy and ideal setup."""
    ideal_estimator = AerEstimator()
    ideal_result = ideal_estimator.run([circuit], [observable])
    ideal_exp_value = ideal_result.result().values[0]

    noisy_estimator = AerEstimator()
    noisy_simulator = AerSimulator().from_backend(backend)
    noisy_estimator._backend = noisy_simulator
    noisy_result = noisy_estimator.run([circuit], [observable])
    noisy_exp_value = noisy_result.result().values[0]
    return ideal_exp_value, noisy_exp_value


def create_meas_data_from_estimators(
    circuits: QuantumCircuit,
    observables: SparsePauliOp,
    estimators: List[BaseEstimator],
    **run_params,
):
    results = []
    for estimator in estimators:
        result = estimator.run(circuits, observables, **run_params).result()
        results.append(result.values[0])
    return results


def encode_pauli_sum_op(op: Union[PauliSumOp, SparsePauliOp]):
    """Encodes pauli sum operator

    Args:
        op: operator

    Returns:
        encoded representation of operator
    """

    if isinstance(op, SparsePauliOp):
        op = PauliSumOp.from_list(op.to_list())

    mapping = {
        "X": [0, 0, 0, 1],
        "Y": [0, 0, 1, 0],
        "Z": [0, 1, 0, 0],
        "I": [1, 0, 0, 0],
    }
    coeffs = [k.coeffs[0].real for k in op]
    strings = [str(k.primitive.paulis[0]) for k in op]
    rows = []
    for c, pauli in zip(coeffs, strings):
        encoded_row = [c]
        for p in pauli:
            encoded_row += mapping.get(p, [0, 0, 0, 0])
        rows.append(encoded_row)
    return rows


def generate_random_pauli_sum_op(
    n_qubits: int, size: int, coeff: Optional[float] = None
) -> PauliSumOp:
    """Generates random pauli sum op."""
    paulis = []
    coeffs = (
        [coeff] * size
        if coeff
        else np.random.uniform(low=-1.0, high=1.0, size=(size,)).tolist()
    )
    for coefficient, pauli in zip(
        coeffs, random_pauli_list(n_qubits, size, phase=False)
    ):
        paulis.append((str(pauli), coefficient))
    return PauliSumOp.from_list(paulis)



if __name__ == '__main__':
    qc = QuantumCircuit(2)
    qc.x(1)
    qc.x(0)
    qc.barrier()
    qc.rz(0.4, [0, 1])
    qc.measure_all()

    from qiskit.providers.fake_provider import FakeLima
    backend = FakeLima()
    circuit_to_graph_data_json(qc, get_backend_properties_v1(backend))
