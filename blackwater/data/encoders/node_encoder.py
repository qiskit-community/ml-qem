"""Node encoders module."""
from abc import ABC
from dataclasses import dataclass
from typing import Union, List, Dict, Optional

from qiskit.circuit import Qubit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.dagcircuit import DAGNode, DAGOpNode
from qiskit.providers import BackendV1, BackendV2, QubitProperties
from qiskit.transpiler import Target

from blackwater.exception import BlackwaterException

N_QUBIT_PROPERTIES = 2
ALL_INSTRUCTIONS = list(get_standard_gate_name_mapping().keys())


class NodeEncoder(ABC):
    """Base class for circuit dag node encoder."""

    def encode(self, node: DAGNode) -> List[float]:
        """Encodes node of circuit dag."""
        raise NotImplementedError


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

    def encode(self, node: DAGNode) -> List[float]:
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
        raise BlackwaterException(
            f"Backend of type [{type(backend)}] is not supported yet."
        )


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

    def encode(self, node: DAGNode) -> List[float]:
        if isinstance(node, DAGOpNode):
            params_encoding = [0.0, 0.0, 0.0]
            for i, param in enumerate(node.op.params):
                if isinstance(param, (float, int)):
                    params_encoding[i] = float(param)
                elif param.is_real():
                    params_encoding[i] = float(param._symbol_expr)

            if node.op.name not in self.encoding_map:
                raise BlackwaterException(
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
