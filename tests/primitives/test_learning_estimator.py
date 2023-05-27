"""Learning estimator tests."""

from unittest import TestCase

import torch
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Estimator, BaseEstimator, EstimatorResult
from qiskit.providers.fake_provider import FakeLimaV2
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from blackwater.data import DefaultNumpyEstimatorInputEncoder
from blackwater.data.encoders.graph_utils import DefaultPyGEstimatorEncoder
from blackwater.data.encoders.utils import generate_random_pauli_sum_op
from blackwater.primitives.learning_esimator import (
    learning_estimator,
    ScikitLearnEstimatorModel,
    TorchGeometricEstimatorModel,
)


# pylint: disable=unused-argument
class DummyModel(torch.nn.Module):
    """Dummy model for tests."""

    def forward(
        self, circuit_nodes, edges, noisy_exp_value, observable, circuit_depth, batch
    ):
        """Forward pass."""
        return noisy_exp_value


class TestLearningEstimator(TestCase):
    """TestLearningEstimator."""

    def test_learning_estimator(self):
        """Test for learning estimator."""
        circuit = random_circuit(5, 2)
        operator = generate_random_pauli_sum_op(5, 2)
        backend = FakeLimaV2()

        data, labels = make_regression(  # pylint: disable=unbalanced-tuple-unpacking
            n_features=48, n_informative=2, random_state=0, shuffle=False
        )
        regressor = RandomForestRegressor(max_depth=2, random_state=0)
        regressor.fit(data, labels)
        model = ScikitLearnEstimatorModel(
            regressor, DefaultNumpyEstimatorInputEncoder()
        )

        estimator: BaseEstimator = learning_estimator(
            Estimator, model=model, backend=backend
        )()
        result = estimator.run(circuit, operator).result()
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.shape, (1,))

    def test_pyg_learning_estimator(self):
        """Tests torch geometric learning estimator."""
        model = TorchGeometricEstimatorModel(DummyModel(), DefaultPyGEstimatorEncoder())
        backend = FakeLimaV2()

        circuit = transpile(random_circuit(5, 2, measure=False), backend)
        obs = generate_random_pauli_sum_op(5, 1, coeff=1.0)

        estimator: BaseEstimator = learning_estimator(
            Estimator, model=model, backend=backend
        )()
        result = estimator.run(circuit, obs).result()
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.shape, (1,))

    # def test_vqe_with_estimator(self):
    #     """Tests estimator in VQE."""
    #     model = DummyModel()
    #     lima = FakeLimaV2()
    #
    #     operator = (
    #         (-1.052373245772859 * I ^ I)
    #         + (0.39793742484318045 * I ^ Z)
    #         + (-0.39793742484318045 * Z ^ I)
    #         + (-0.01128010425623538 * Z ^ Z)
    #         + (0.18093119978423156 * X ^ X)
    #     )
    #
    #     ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
    #
    #     options = Options(optimization_level=0)
    #     ngem_estimator = ngem(Estimator, model=model, backend=lima, options=options)
    #     estimator = ngem_estimator()
    #
    #     slsqp = SLSQP(maxiter=1)
    #     vqe = VQE(estimator, ansatz, slsqp)
    #     result = vqe.compute_minimum_eigenvalue(operator)
    #     self.assertIsInstance(result, VQEResult)
