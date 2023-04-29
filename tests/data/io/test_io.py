"""Tests for io."""
import os
import shutil
from pathlib import Path
from unittest import TestCase

from qiskit import transpile
from qiskit.providers.fake_provider import FakeLimaV2
from qiskit.quantum_info import SparsePauliOp

from blackwater.data.dataclasses import ExpValData
from blackwater.data.io.io import ExpValDataWriter, ExpValDataReader
from tests.data.encoders.test_graph import create_bell_circuit


class TestExpValIO(TestCase):
    """TestExpValIO."""

    def setUp(self) -> None:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        resource_dir = os.path.join(current_dir, "..", "..", "resources")
        self.data_folder = os.path.join(resource_dir, "data")
        Path(self.data_folder).mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if os.path.exists(self.data_folder):
            shutil.rmtree(self.data_folder)

    def test_io(self):
        """Tests io."""
        circuit = create_bell_circuit(3)
        fake_lima = FakeLimaV2()
        transpiled_circuit = transpile(circuit, fake_lima)

        exp_val_data = ExpValData.build(
            circuit=transpiled_circuit,
            expectation_values=[0.0],
            observable=SparsePauliOp(["ZZZ"]),
            backend=fake_lima,
        )

        file_path = os.path.join(self.data_folder, "test_data.json")

        writer = ExpValDataWriter()
        writer.save_to_file(path=file_path, data=[exp_val_data])

        reader = ExpValDataReader()
        loaded_data = reader.read_from_file(file_path)
        self.assertIsInstance(loaded_data[0], ExpValData)
