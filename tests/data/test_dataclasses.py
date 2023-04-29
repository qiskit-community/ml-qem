"""Tests fpr dataclasses."""

from unittest import TestCase

from qiskit import transpile
from qiskit.providers.fake_provider import FakeLimaV2
from qiskit.quantum_info import SparsePauliOp
from torch_geometric.data import Data

from blackwater.data.dataclasses import ExpValData
from tests.data.encoders.test_graph import create_bell_circuit


class TestDataClasses(TestCase):
    """TestDataclasses."""

    def test_exp_val_base(self):
        """Tests exp val dataclass."""
        circuit = create_bell_circuit(3)
        fake_lima = FakeLimaV2()
        transpiled_circuit = transpile(circuit, fake_lima)

        exp_val_data = ExpValData.build(
            circuit=transpiled_circuit,
            expectation_values=[0.0],
            observable=SparsePauliOp(["ZZZ"]),
            backend=fake_lima,
        )
        self.assertIsInstance(exp_val_data, ExpValData)

        pyg_data = exp_val_data.to_pyg()
        self.assertIsInstance(pyg_data, Data)
        self.assertEqual(list(pyg_data.x.shape), [9, 22])
        self.assertEqual(list(pyg_data.edge_index.shape), [2, 10])
        self.assertEqual(list(pyg_data.edge_attr.shape), [10, 1])
