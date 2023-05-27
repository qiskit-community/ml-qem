"""Operator encoders."""

import numpy as np
from qiskit.quantum_info import Operator

from blackwater.data.core import OperatorEncoder
from blackwater.data.encoders.utils import encode_operator


class DefaultOperatorEncoder(OperatorEncoder):
    """Default operator encoder to turn operator class into numpy array."""

    def encode(self, operator: Operator, **kwargs) -> np.ndarray:  # type: ignore
        """Encodes operator.

        Args:
            operator: operator to encoder
            **kwargs: other arguments

        Returns:
            numpy array
        """
        operator_encoding = []
        for entry in encode_operator(operator).operator:
            operator_encoding += entry

        return np.array(operator_encoding)
