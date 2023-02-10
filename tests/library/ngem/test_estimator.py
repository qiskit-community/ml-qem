"""Tests for NGEM estimator."""

from unittest import TestCase

import torch
from qiskit import transpile
from qiskit.algorithms.minimum_eigensolvers import VQE, VQEResult
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.random import random_circuit
from qiskit.opflow import I, X, Z
from qiskit.primitives import Estimator, EstimatorResult
from qiskit.providers import Options
from qiskit.providers.fake_provider import FakeLima

from blackwater.data.utils import generate_random_pauli_sum_op
from blackwater.library.ngem.estimator import ngem


# pylint: disable=no-member,no-value-for-parameter,unused-argument
class DummyModel(torch.nn.Module):
    """Dummy model for tests."""

    def forward(self, exp_value, observable, circuit_depth, nodes, edge_index, batch):
        """Forward pass."""
        return exp_value


class TestEstimator(TestCase):
    """Test Ngem estimator."""

    def test_estimator(self):
        """Tests estimator."""
        model = DummyModel()
        lima = FakeLima()

        circuit = transpile(random_circuit(5, 2, measure=False), lima)
        obs = generate_random_pauli_sum_op(5, 1, coeff=1.0)

        ngem_estimator = ngem(Estimator, model=model, backend=lima)
        estimator = ngem_estimator()
        job = estimator.run([circuit], [obs])
        result = job.result()

        self.assertIsInstance(result, EstimatorResult)

    def test_vqe_with_estimator(self):
        """Tests estimator in VQE."""
        model = DummyModel()
        lima = FakeLima()

        operator = (
            (-1.052373245772859 * I ^ I)
            + (0.39793742484318045 * I ^ Z)
            + (-0.39793742484318045 * Z ^ I)
            + (-0.01128010425623538 * Z ^ Z)
            + (0.18093119978423156 * X ^ X)
        )

        ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

        options = Options(optimization_level=0)
        ngem_estimator = ngem(Estimator, model=model, backend=lima, options=options)
        estimator = ngem_estimator()

        slsqp = SLSQP(maxiter=1)
        vqe = VQE(estimator, ansatz, slsqp)
        result = vqe.compute_minimum_eigenvalue(operator)
        self.assertIsInstance(result, VQEResult)
