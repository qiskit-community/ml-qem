"""Tests for exp value dataloader."""
import json
import os
from unittest import TestCase

from qiskit.providers.fake_provider import FakeLima
from torch_geometric.data import Data

from blackwater.data.generators.exp_val import exp_value_generator
from blackwater.data.loaders.exp_val import CircuitGraphExpValMitigationDataset


class TestExpValDataloader(TestCase):
    """TestExpValDataloader."""

    def setUp(self) -> None:
        self.file_name = "entries.json"
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.path_to_file = os.path.join(self.current_dir, self.file_name)

        lima = FakeLima()

        generator = exp_value_generator(
            backend=lima, n_qubits=5, circuit_depth=2, pauli_terms=1, max_entries=2
        )

        with open(  # pylint: disable=unspecified-encoding
            self.path_to_file, "w"
        ) as entries_file:
            json.dump([entry.to_dict() for entry in generator], entries_file)

    def tearDown(self) -> None:
        if os.path.exists(self.path_to_file):
            os.remove(self.path_to_file)

    def test_dataloader(self):
        """Tests dataloader."""
        dataset = CircuitGraphExpValMitigationDataset([self.path_to_file])

        for data in dataset:
            self.assertIsInstance(data, Data)
        self.assertEqual(len(dataset), 2)
