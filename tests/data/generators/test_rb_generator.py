"""Tests for RB generator."""
from unittest import TestCase

from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.providers.fake_provider import FakeLima

from blackwater.data.generators.exp_val import ExpValueEntry
from blackwater.data.generators.rb import rb_generator


class TestRBGenerator(TestCase):
    """TestRBGenerator."""

    def test_generator(self):
        """Tests rb generator."""
        lima = FakeLima()
        gen = rb_generator(
            max_iter=1, backend=lima, n_qubits=5, num_samples=1, lengths=[1]
        )

        counter = 0
        for entry, circuit, observable in gen:
            self.assertIsInstance(circuit, QuantumCircuit)
            self.assertIsInstance(observable, PauliSumOp)
            self.assertIsInstance(entry, ExpValueEntry)

            counter += 1
        self.assertEqual(counter, 1)
