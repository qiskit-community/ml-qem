"""Numpy encoder."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp

from blackwater.data.core import DataEncoder
from blackwater.data.encoders.circuit import DefaultCircuitEncoder
from blackwater.data.encoders.operator import DefaultOperatorEncoder


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

        circuit_encoding = DefaultCircuitEncoder().encode(circuit)
        operator_encoding = DefaultOperatorEncoder().encode(operator)

        return np.concatenate([[exp_value], circuit_encoding, operator_encoding])
