"""Agent class for RL."""
from abc import ABC
from dataclasses import dataclass

from torch.nn import Module


Model = Module


@dataclass
class ActionResult:
    """Action result."""

    reward: float
    score: float
    done: bool


class Action(ABC):
    """Base class for actions."""


class State(ABC):
    """Base class for state."""


class Environment(ABC):
    """Environment class for RL algorithms,"""

    def perform_action(self, action: Action) -> ActionResult:
        """Performs action on environment."""
        raise NotImplementedError

    def get_state(self) -> State:  # type: ignore
        """Returns state of environment."""
        raise NotImplementedError


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

    def perform_action(self, action: Action) -> ActionResult:
        """Performs action."""
        return self.env.perform_action(action)
