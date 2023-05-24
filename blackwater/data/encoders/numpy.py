"""Numpy encoder."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp

from blackwater.data.core import DataEncoder
from blackwater.data.encoders.utils import encode_operator


class DefaultNumpyEstimatorInputEncoder(DataEncoder):
    """Standard encoder for learning estimator that transforms
    circuit and observables into numpy array.
    """

    def encode(self, **kwargs):
        """Encodes estimator data into numpy array

        Args:
            **kwargs: circuit and operator

        Returns:
            numpy array where:
            - 0th element - noisy exp value
            - first element - depth of circuit
            - second element - 2q depth of circuit
            - 3rd - number of 1q gates
            - 4th - number of 2q gates
            - 5th - num qubits
            - rest - encoded operator
        """
        circuit: QuantumCircuit = kwargs.get("circuit")
        operator: PauliSumOp = kwargs.get("operator")
        exp_value = kwargs.get("exp_val")

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

        circuit_encoding = [
            depth,
            two_qubit_depth,
            num_one_q_gates,
            num_two_q_gates,
            circuit.num_qubits,
        ]

        operator_encoding = []
        for entry in encode_operator(operator).operator:
            operator_encoding += entry

        data_encoding = [exp_value] + circuit_encoding + operator_encoding
        return np.array(data_encoding)
