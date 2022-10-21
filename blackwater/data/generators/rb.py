"""Randomised benchmarking data generator."""
import random
from typing import Optional, List, Tuple

from qiskit import transpile, QuantumCircuit
from qiskit.providers import BackendV1
from qiskit.quantum_info import Operator
from qiskit_experiments.library import StandardRB

from blackwater.data.generators.exp_val import ExpValueEntry
from blackwater.data.utils import (
    circuit_to_graph_data_json,
    generate_random_pauli_sum_op,
    get_backend_properties_v1,
    create_estimator_meas_data,
    encode_pauli_sum_op,
)


def generate_rb_circuit(
    n_qubits: int,
    backend: Optional[BackendV1] = None,
    num_samples: int = 1,
    lengths: List[int] = None,
):
    """Generates RB circuit.

    Args:
        n_qubits: num qubits
        backend: backend to transpile for if given
        num_samples: num samples for RB
        lengths: length for RB

    Returns:
        RB circuit
    """
    lengths = lengths or [5]
    exp = StandardRB(qubits=range(n_qubits), lengths=lengths, num_samples=num_samples)
    circuit = exp.circuits()[0]
    if backend:
        circuit = transpile(circuit, backend, optimization_level=0)
    return circuit


def rb_generator(
    backend: BackendV1,
    n_qubits: int,
    max_iter: int = 1000,
    pauli_terms: int = 1,
    pauli_coeff: float = 1.0,
    lengths: Optional[List[int]] = None,
    num_samples: Optional[int] = None,
) -> Tuple[ExpValueEntry, QuantumCircuit, Operator]:
    """Generator for RB circuits.

    Args:
        backend: backend to run tranpile for
        n_qubits: number of qubits
        max_iter: max ites for generator
        pauli_terms: number of Puali terms in generated observable
        pauli_coeff: Pauli coefficient
        lengths: lengths for RB generation
        num_samples: num samples for RB

    Returns:

    """
    properties = get_backend_properties_v1(backend)

    for _ in range(max_iter):
        if num_samples is None:
            num_samples = random.randint(1, 3)

        circuit = generate_rb_circuit(
            backend=backend, n_qubits=n_qubits, num_samples=num_samples, lengths=lengths
        )

        graph_data = circuit_to_graph_data_json(
            circuit=circuit,
            properties=properties,
            use_qubit_features=True,
            use_gate_features=True,
        )
        observable = generate_random_pauli_sum_op(n_qubits, pauli_terms, pauli_coeff)

        ideal_exp_val, noisy_exp_val = create_estimator_meas_data(
            backend=backend, circuit=circuit, observable=observable
        )

        yield ExpValueEntry(
            circuit_graph=graph_data,
            observable=encode_pauli_sum_op(observable),
            ideal_exp_value=ideal_exp_val,
            noisy_exp_value=noisy_exp_val,
            circuit_depth=circuit.depth(),
        ), circuit, observable
