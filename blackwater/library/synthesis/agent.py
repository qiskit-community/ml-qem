"""Synthesis agent."""
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.extensions import UnitaryGate
from qiskit.providers import Backend
from qiskit.quantum_info import Operator
from qiskit_aer.backends.aerbackend import AerBackend
from qiskit_aer import Aer

from blackwater.rl.agent import Agent, Action, ActionResult, Environment, State


@dataclass
class AppendGateToCircuitAction(Action):
    """AppendGateToCircuitAction."""
    gate: Gate
    qubits: List[int]


@dataclass
class SynthesisEnvironmentState(State):
    """SynthesisEnvironmentState."""
    circuit: QuantumCircuit
    unitary: Operator


class SynthesisEnvironment(Environment):
    """SynthesisEnvironment."""

    def __init__(
        self,
        n_qubits: int,
        unitary: Union[np.ndarray, Operator],
        shots: int = 1000,
        max_iter: int = 100
    ):
        self.n_qubits = n_qubits
        self.target_unitary = unitary
        self.shots = shots
        self.max_iter = max_iter
        self.current_iteration = 0

        self.circuit = QuantumCircuit(n_qubits)
        self.backend: AerBackend = Aer.get_backend('unitary_simulator')

    def perform_action(self, action: AppendGateToCircuitAction) -> ActionResult:
        self.current_iteration += 1

        self.circuit.append(action.gate, action.qubits)

        result = self.backend.run(self.circuit, shots=self.shots).result()
        unitary = result.get_unitary(self.circuit, 3)
        reward = np.linalg.norm(self.target_unitary.data - unitary.data)

        done = np.isclose(reward, 0.0) or self.current_iteration > self.max_iter

        # TODO: figure out score
        return ActionResult(reward=-reward, score=0, done=done)

    def get_state(self):
        return SynthesisEnvironmentState(self.circuit, self.target_unitary)
