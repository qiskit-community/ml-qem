"""Agent class for RL."""
from dataclasses import dataclass

from torch.nn import Module

from blackwater.rl.env import Environment


@dataclass
class ActionResult:
    """Action result."""

    reward: float
    score: float
    done: bool


class Agent:
    """Agent class for RL."""

    def __init__(self, env: Environment, model: Module):
        self.env = env
        self.model = model

    def select_action(self, *args, **kwargs):
        """Returns action based on state of env."""
        raise NotImplementedError

    def optimize_model(self, *args, **kwargs):
        """Optimization step."""
        raise NotImplementedError

    def perform_action(self, action) -> ActionResult:
        """Performs action."""
        raise NotImplementedError
