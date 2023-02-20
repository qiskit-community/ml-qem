"""Tests for data utils."""
from unittest import TestCase

from qiskit import QuantumCircuit
from torch_geometric.data import Data

from blackwater.data.utils import circuit_to_pyg_data


class TestDataUtils(TestCase):
    """Tests for data utilities."""

    def test_circuit_to_data(self):
        """Test for circuit_to_data function."""

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        data = circuit_to_pyg_data(circuit)

        self.assertIsInstance(data, Data)
        self.assertEqual(data.x.shape, (5, 34))
        self.assertEqual(data.edge_index.shape, (2, 5))


if __name__ == '__main__':
    TestDataUtils().test_circuit_to_data()