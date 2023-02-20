"""Test for exp val generator."""
from unittest import TestCase

from qiskit.providers.fake_provider import FakeLima
from torch_geometric.data import Data

from blackwater.data.generators.exp_val import exp_value_generator, ExpValueEntry


class TestExpValDataGenerator(TestCase):
    """TestExpValDataGenerator."""

    def test_generator(self):
        """Tests generator."""
        lima = FakeLima()

        generator = exp_value_generator(
            backend=lima, n_qubits=5, circuit_depth=2, pauli_terms=1, max_entries=3
        )

        entries = list(generator)
        self.assertEqual(len(entries), 3)
        self.assertIsInstance(entries[0], ExpValueEntry)
        self.assertIsInstance(entries[0].to_pyg_data(), Data)


if __name__ == '__main__':
    TestExpValDataGenerator().test_generator()
