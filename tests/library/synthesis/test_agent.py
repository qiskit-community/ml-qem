"""Tests agent."""
from unittest import TestCase

from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, XGate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Operator

from blackwater.library.synthesis.agent import SynthesisEnvironment, AppendGateToCircuitAction


class TestSynthesisAgent(TestCase):
    """TestSynthesisAgent."""

    def test_env(self):
        """Tests synthesis environment,"""
        matrix = [[0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0]]
        env = SynthesisEnvironment(
            n_qubits=2,
            unitary=Operator(matrix)
        )
        result = env.perform_action(
            action=AppendGateToCircuitAction(gate=XGate(), qubits=[1])
        )
        self.assertEqual(result.reward, -2.0)
