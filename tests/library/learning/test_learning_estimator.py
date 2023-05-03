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
from qiskit.providers.fake_provider import FakeLimaV2, FakeLima, ConfigurableFakeBackend

from blackwater.data.utils import generate_random_pauli_sum_op, get_backend_properties_v1
from blackwater.library.learning.estimator import learning, EmptyProcessor, TorchLearningModelProcessor
from blackwater.library.temp import encode_data, MLP1


class TestLearningEstimator(TestCase):
    """Test learning estimator."""

    def test_temp(self):
        circuits = [
            random_circuit(3, 2)
            for _ in range(4)
        ]
        ideal_exp_values = [[0.] for _ in range(4)]
        noisy_exp_vals = [[0.] for _ in range(4)]

        backend = FakeLima()
        properties = get_backend_properties_v1(backend)

        data = encode_data(
            circuits=circuits,
            properties=properties,
            ideal_exp_vals=ideal_exp_values,
            noisy_exp_vals=noisy_exp_vals,
            num_qubits=len(ideal_exp_values[0])
        )
        print(data)

    def test_torch_estimator(self):
        """Test torch estimator with learning model."""

        backend = ConfigurableFakeBackend("Tashkent", n_qubits=2, version=1)
        model = MLP1(63, 5, 1)

        processor = TorchLearningModelProcessor(
            model=model,
            backend=backend
        )

        circuit = transpile(random_circuit(2, 2, measure=False), backend)
        obs = generate_random_pauli_sum_op(2, 1, coeff=1.0)

        ngem_estimator = learning(Estimator, processor=processor, backend=backend, skip_transpile=True)
        estimator = ngem_estimator()
        job = estimator.run([circuit], [obs])
        result = job.result()

        self.assertIsInstance(result, EstimatorResult)

    def test_lre_with_vqe(self):
        """Test learning based estimator with VQE."""
        backend = ConfigurableFakeBackend("Tashkent", n_qubits=2, version=1)
        model = MLP1(63, 5, 1)

        processor = TorchLearningModelProcessor(
            model=model,
            backend=backend
        )

        circuit = transpile(random_circuit(2, 2, measure=False), backend)
        obs = generate_random_pauli_sum_op(2, 1, coeff=1.0)

        operator = (
            (-1.052373245772859 * I ^ I)
            + (0.39793742484318045 * I ^ Z)
            + (-0.39793742484318045 * Z ^ I)
            + (-0.01128010425623538 * Z ^ Z)
            + (0.18093119978423156 * X ^ X)
        )

        coefficent = [-1.052373245772859]
        paulis = [I ^ I, ]

        ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

        options = Options(optimization_level=0)
        ngem_estimator = learning(Estimator, processor=processor, backend=backend, skip_transpile=True)
        estimator = ngem_estimator()

        slsqp = SLSQP(maxiter=1)
        vqe = VQE(estimator, ansatz, slsqp)
        result = vqe.compute_minimum_eigenvalue(operator)
        self.assertIsInstance(result, VQEResult)



    def test_estimator(self):
        """Tests estimator."""
        processor = EmptyProcessor()
        lima = FakeLimaV2()

        circuit = transpile(random_circuit(5, 2, measure=False), lima)
        obs = generate_random_pauli_sum_op(5, 1, coeff=1.0)

        ngem_estimator = learning(Estimator, processor=processor, backend=lima, skip_transpile=True)
        estimator = ngem_estimator()
        job = estimator.run([circuit], [obs])
        result = job.result()

        self.assertIsInstance(result, EstimatorResult)

    def test_vqe_with_estimator(self):
        """Tests estimator in VQE."""
        model = DummyModel()
        lima = FakeLimaV2()

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
