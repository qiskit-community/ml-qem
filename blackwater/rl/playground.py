from __future__ import annotations

import numpy as np
from typing import Optional, List, SupportsFloat, Any, Tuple

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import HGate


def circuit_to_observation(circuit: QuantumCircuit) -> ObsType:
    """Converts circuit to observation

    Args:
        circuit: circuit to convert

    Returns:
        observation
    """
    pass


def action_to_circuit_instruction(action: ActType) -> Tuple[Instruction, List[int], List[int]]:
    """Converts action to instruction to append

    Args:
        action: action

    Returns:
        instruction and set or registers
    """
    pass


class QuantumCircuitEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 8
    }

    def __init__(
        self,
        n_qubits: int,
        n_instructions: Optional[int] = None,
        instruction_set: Optional[List[Instruction]] = None,
        render_mode: Optional[str] = None
    ):
        # circuit
        # list / tuple
        # - instruction type - discrete
        # - qubits acts on - discrete
        # - parameters - box
        # -

        self.n_qubits = n_qubits
        self.n_instructions = n_instructions or 100
        self.instruction_set = instruction_set or [HGate()]
        self.render_mode = render_mode

        self.circuit = QuantumCircuit(n_qubits)

        self.action_space = spaces.Dict({
            "instruction_type": spaces.Discrete(len(instruction_set)),
            "qubits_acts_on": spaces.Box(0, 1, shape=(self.n_qubits,), dtype=np.int),
            "parameters": spaces.Box(0, 1, shape=(3,), dtype=np.float)
        })

        self.observation_space = spaces.Dict({
            "instruction_types": spaces.Box(0, 1, shape=(len(instruction_set), self.n_instructions)),
            "acting_qubits": spaces.Box(0, 1, shape=(self.n_qubits, self.n_instructions)),
            "parameters": spaces.Box(0, 1, shape=(3, self.n_instructions), dtype=np.float)
        })

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        instr, qreg, _ = action_to_circuit_instruction(action)
        self.circuit.append(instr, qreg)

        observation = None
        reward = None
        done = None

        return observation, reward, done, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        return


def run():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == '__main__':
    run()
