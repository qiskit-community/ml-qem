"""Tests for numpy encoders."""

from unittest import TestCase

import numpy as np
from qiskit.circuit.random import random_circuit

from blackwater.data import DefaultNumpyEstimatorInputEncoder
from blackwater.data.encoders.utils import generate_random_pauli_sum_op


class TestNumpyEncoders(TestCase):
    """Tests for numpy encoders."""

    def test_numpy_estimator_encoder(self):
        """Tests numpy estimator encoder."""
        encoder = DefaultNumpyEstimatorInputEncoder()

        circuit = random_circuit(3, 2)
        operator = generate_random_pauli_sum_op(3, 2)

        result = encoder.encode(circuit=circuit, operator=operator)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (32,))
