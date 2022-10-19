"""NGS."""
from ctypes import Union
from typing import List

from torch import Tensor

from blackwater.rl.agent import Agent, ActionResult
from blackwater.rl.env import Environment
from .model import NGSModel


class NGSEnvironment(Environment):
    """NGS environment."""

    def __init__(self):
        pass

    def get_state(self) -> Union[Tensor, List[Tensor]]:  # type: ignore
        pass


class NGSAgent(Agent):
    """NGS agent."""

    def __init__(self):
        model = NGSModel()
        env = NGSEnvironment()

        super().__init__(env, model)

    def optimize_model(self, *args, **kwargs):
        pass

    def perform_action(self, action) -> ActionResult:
        pass

    def select_action(self, *args, **kwargs):
        pass
