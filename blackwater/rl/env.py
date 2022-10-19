"""Environment for RL."""
from abc import ABC
from ctypes import Union
from typing import List

from torch import Tensor


class Environment(ABC):
    """Environment class for RL algorithms,"""

    def get_state(self) -> Union[Tensor, List[Tensor]]:  # type: ignore
        """Returns state of environment."""
        raise NotImplementedError
