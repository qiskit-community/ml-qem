"""Tests for NGEM estimator."""

from unittest import TestCase

import torch
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Estimator, EstimatorResult
from qiskit.providers.fake_provider import FakeLima

from blackwater.data.utils import generate_random_pauli_sum_op
from blackwater.library.ngem.estimator import ngem


class DummyModel(torch.nn.Module):
    def forward(self, exp_value, observable, circuit_depth, nodes, edge_index, batch):
        return exp_value


class TestEstimator(TestCase):
    """Test Ngem estimator."""

    def test_estimator(self):
        """Tests estimator."""
        model = DummyModel()
        lima = FakeLima()

        circuit = transpile(random_circuit(5, 2, measure=False), lima)
        obs = generate_random_pauli_sum_op(5, 1, coeff=1.0)

        NgemEstimator = ngem(Estimator, model=model, backend=lima)
        estimator = NgemEstimator()
        result = estimator.run([circuit], [obs]).result()

        self.assertIsInstance(result, EstimatorResult)
