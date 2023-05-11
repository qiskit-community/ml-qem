"""Circuit builder Gym Env."""

from __future__ import annotations

import time
from typing import List, SupportsFloat, Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from matplotlib import pyplot as plt  # pylint: disable=import-error
from qiskit import QuantumCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping


# pylint: disable=attr-defined,arg-type
def _fix_qargs(qargs: List[int], num_qubits: int):
    seen_qubits: Dict[int, List[int]] = {}
    for idx, qarg in enumerate(qargs):
        if qarg in seen_qubits:
            seen_qubits[qarg].append(idx)
        else:
            seen_qubits[qarg] = [idx]

    duplicates = [qarg for qarg, idxs in seen_qubits.items() if len(idxs) > 1]
    if len(duplicates) > 0:
        duplicate_idxs = []
        for qarg, idxs in seen_qubits.items():
            if len(idxs) > 1:
                for idx in idxs[1:]:
                    duplicate_idxs.append(idx)

        max_of_duplicates = max(duplicates)
        available_qargs = [qarg for qarg in range(num_qubits) if qarg not in qargs]
        for idx in duplicate_idxs:
            qargs[idx] = available_qargs[0]
            del available_qargs[0]
            max_of_duplicates += 1

    return qargs


class QuantumCircuitBuilderEnv(gym.Env):
    """Base quantum circuit builder gymnasium env."""

    metadata = {"render_modes": ["human", "console"], "render_fps": 8}

    def __init__(self, env_config: Dict[str, Any]):
        n_qubits = env_config.get("n_qubits")
        model = env_config.get("model")
        instruction_set = env_config.get("instruction_set")
        render_mode = env_config.get("render_mode")

        self.n_qubits = n_qubits or 2
        self.instruction_set = instruction_set or list(
            get_standard_gate_name_mapping().values()
        )
        self.n_instructions = 10

        self.render_mode = render_mode or "human"

        self.circuit = QuantumCircuit(n_qubits)

        self.model = model

        self.action_space = spaces.Dict(
            {
                "instruction_type": spaces.Discrete(len(self.instruction_set)),
                "qubits_acts_on": spaces.Box(
                    0, self.n_qubits - 1, shape=(3,), dtype=int
                ),
                "parameters": spaces.Box(0, 1, shape=(4,), dtype=float),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "instruction_types": spaces.Box(
                    0,
                    len(self.instruction_set),
                    shape=(len(self.instruction_set), self.n_instructions),
                    dtype=int,
                ),
                "acting_qubits": spaces.Box(
                    0, self.n_qubits - 1, shape=(3, self.n_instructions), dtype=int
                ),
                "parameters": spaces.Box(
                    0, 1, shape=(4, self.n_instructions), dtype=float
                ),
            }
        )

        self.reward_history: List[float] = []

        if render_mode == "human":
            fig = plt.figure()
            self.ax_circuit = fig.add_subplot(211)
            ax_histories = fig.add_subplot(212)
            ax_histories.set_ylabel("Reward")
            ax_histories.set_xlabel("steps")
            self.ax_histories = ax_histories

            plt.show(block=False)

    def _apply_action_to_circuit(self, action: ActType) -> QuantumCircuit:
        """Converts action to instruction to append

        Args:
            action: action

        Returns:
            instruction and set or registers
        """
        instruction = self.instruction_set[action.get("instruction_type")]
        qargs = action.get("qubits_acts_on").tolist()[: instruction.num_qubits]
        params = action.get("parameters").tolist()[: len(instruction.params)]
        instruction.params = params
        qargs = _fix_qargs(qargs, self.n_qubits)
        self.circuit.append(instruction, qargs)
        return self.circuit

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Environment step

        Returns:
            observation (ObsType), reward (float), done (bool), False, info (dict)
        """
        raise NotImplementedError

    def _get_empty_observation(self):
        """Returns empty observation space (reset)."""
        return {
            "instruction_types": np.zeros(
                (len(self.instruction_set), self.n_instructions)
            ).astype(int),
            "acting_qubits": np.zeros((3, self.n_instructions)).astype(int),
            "parameters": np.zeros((4, self.n_instructions)),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.circuit = QuantumCircuit(self.n_qubits)
        return self._get_empty_observation(), {}

    def render(self):
        if self.render_mode == "human":
            self.ax_circuit.clear()
            self.ax_histories.clear()
            self.circuit.draw("mpl", ax=self.ax_circuit)
            self.ax_histories.plot(range(len(self.reward_history)), self.reward_history)
            time.sleep(0.3)
            plt.pause(0.001)

        elif self.render_mode == "console":
            print(self.circuit)
