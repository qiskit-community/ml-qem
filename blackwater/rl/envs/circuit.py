"""Circuit builder Gym Env."""

from __future__ import annotations

import time
from typing import List, SupportsFloat, Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from matplotlib import pyplot as plt  # pylint: disable=import-error
from numpy import integer, floating
from qiskit import QuantumCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Operator


def _fix_qargs(qargs: List[int], num_qubits: int):
    """Fixes arguments for qubits."""
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
        self.n_instructions = env_config.get("n_instructions", 10)

        self.render_mode = render_mode or "human"

        self.circuit = QuantumCircuit(n_qubits)
        self.observation = self._get_empty_observation()

        self.model = model
        self.instructions_added = 0

        self.action_space = spaces.Dict(
            {
                "instruction_type": spaces.Discrete(len(self.instruction_set)),
                "qubits_acts_on": spaces.Box(
                    0, self.n_qubits - 1, shape=(3,), dtype=integer
                ),
                "parameters": spaces.Box(0, 1, shape=(4,), dtype=floating),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "instruction_types": spaces.Box(
                    0,
                    len(self.instruction_set),
                    shape=(self.n_instructions, len(self.instruction_set)),
                    dtype=integer,
                ),
                "acting_qubits": spaces.Box(
                    0, self.n_qubits - 1, shape=(self.n_instructions, 3), dtype=integer
                ),
                "parameters": spaces.Box(
                    0, 1, shape=(self.n_instructions, 4), dtype=floating
                ),
            }
        )

        # vis specific
        self.reward_history: List[float] = []

        if render_mode == "human":
            fig = plt.figure()
            self.ax_circuit = fig.add_subplot(211)
            ax_histories = fig.add_subplot(212)
            ax_histories.set_ylabel("Reward")
            ax_histories.set_xlabel("steps")
            self.ax_histories = ax_histories

            plt.show(block=False)

    def one_hot_encode_instr_type(self, instruction_type: int):
        """Encodes instruction type into one-hot encoded vector."""
        result = np.zeros(len(self.instruction_set))
        result[instruction_type] = 1
        return result

    def apply_action_to_circuit(self, action: dict):
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
        return params, qargs

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
                (self.n_instructions, len(self.instruction_set))
            ).astype(int),
            "acting_qubits": np.zeros((self.n_instructions, 3)).astype(int),
            "parameters": np.zeros((self.n_instructions, 4)),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.circuit = QuantumCircuit(self.n_qubits)
        self.observation = self._get_empty_observation()
        return self.observation, {}

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


class UnitarySynthesisEnv(QuantumCircuitBuilderEnv):
    """UnitarySynthesisEnv.

    Example:
        >>> instruction_set = [
        >>>     XGate(),
        >>>     CXGate(),
        >>>     HGate(),
        >>>     U3Gate(Parameter("ϴ"), Parameter("φ"), Parameter("λ")),
        >>> ]
        >>> render_mode = "console"
        >>> env_config = {
        >>>     "operator": Operator(random_circuit(2, 3)),
        >>>     "n_qubits": 2,
        >>>     "model": None,
        >>>     "instruction_set": instruction_set,
        >>>     "n_instructions": 20,
        >>>     "render_mode": render_mode
        >>> }
        >>> config = (
        >>>     PPOConfig()
        >>>     .environment(UnitarySynthesisEnv, env_config=env_config)
        >>>     .framework("torch")
        >>> )
        >>>
        >>> result = config.build().train()
    """

    def __init__(self, env_config: Dict[str, Any]):
        """Constructor for env.

        Args:
            env_config: environment configuration
        """
        super().__init__(env_config)
        self.target_unitary = env_config.get("operator").data

    def get_reward(self) -> float:
        """Calculates reward."""
        circuit_unitary = Operator(self.circuit).data
        trace = np.dot(circuit_unitary, np.linalg.inv(self.target_unitary)).trace().real
        ideal_trace = self.n_qubits**2
        return abs(ideal_trace - trace)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        idx = self.instructions_added

        self.observation["instruction_types"][idx] = self.one_hot_encode_instr_type(
            action.get("instruction_type")  # type: ignore
        )
        self.observation["acting_qubits"][idx] = action["qubits_acts_on"]  # type: ignore
        self.observation["parameters"][idx] = action["parameters"]  # type: ignore
        self.apply_action_to_circuit(action)  # type: ignore

        done = idx > self.n_instructions

        return self.observation, self.get_reward(), done, False, {}
