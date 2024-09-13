"""Dataclasses module."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Any

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGNode
from qiskit.providers import Backend
from qiskit.quantum_info import Operator


# pylint: disable=arguments-differ
class DataEncoder:
    """Base data encode class."""

    @abstractmethod
    def encode(self, **kwargs):
        """Encodes data

        Args:
            **kwargs: data to encode

        Returns:
            encoded data
        """
        raise NotImplementedError


class DataDecoder:
    """Base data decoder class."""

    @classmethod
    def decode(cls, data: Any):
        """Decodes from data to object.

        Args:
            data: encoded data

        Returns:
            decoded object
        """
        raise NotImplementedError


class CircuitEncoder(DataEncoder):
    """Base encoder class for circuit objects."""

    @abstractmethod
    def encode(self, circuit: QuantumCircuit, **kwargs):  # type: ignore
        raise NotImplementedError


class OperatorEncoder(DataEncoder):
    """Base encoder class for operator objects."""

    @abstractmethod
    def encode(self, operator: Operator, **kwargs):  # type: ignore
        raise NotImplementedError


class BackendEncoder(DataEncoder):
    """Base encoder class for backend objects."""

    @abstractmethod
    def encode(self, backend: Backend, **kwargs):  # type: ignore
        raise NotImplementedError


class NodeEncoder(DataEncoder):
    """Base class for circuit dag node encoder."""

    def encode(self, node: DAGNode, **kwargs) -> List[float]:  # type: ignore
        """Encodes node of circuit dag."""
        raise NotImplementedError


# pylint: disable=no-member
@dataclass
class MLQEMData:
    """MLQEMData."""

    def serialize(self) -> dict:
        """Serialize class data to dictionary"""
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict):
        """Deserializes data to class."""
        raise NotImplementedError
