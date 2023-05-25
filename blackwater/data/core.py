"""Dataclasses module."""
from dataclasses import dataclass


class DataEncoder:
    """Base data encode class."""

    def encode(self, **kwargs):
        """Encodes data

        Args:
            **kwargs: data to encode

        Returns:
            encoded data
        """
        raise NotImplementedError


# pylint: disable=no-member
@dataclass
class BlackwaterData:
    """BlackwaterData."""

    def serialize(self) -> dict:
        """Serialize class data to dictionary"""
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict):
        """Deserializes data to class."""
        raise NotImplementedError
