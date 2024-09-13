"""IO."""

import json
from typing import List

from ml_qem.data.core import MLQEMData
from ml_qem.data.encoders.graph_utils import ExpValData


# pylint: disable=unspecified-encoding
class Writer:
    """Base writer class."""

    def save_to_file(self, path: str, data: List[MLQEMData]):
        """Writes data to file.

        Args:
            path: path of file to write to
            data: data to write
        """
        raise NotImplementedError


class Reader:
    """Base reader class."""

    def read_from_file(self, path: str) -> List[MLQEMData]:
        """Reads from file.

        Args:
            path: path to file to read from.

        Returns:
            list of data entries
        """
        raise NotImplementedError


class ExpValDataWriter(Writer):
    """ExpValDataWriter."""

    def save_to_file(self, path: str, data: List[MLQEMData]):
        with open(path, "w") as file:
            json.dump([entry.serialize() for entry in data], file, indent=4)


class ExpValDataReader(Reader):
    """ExpValDataReader."""

    def read_from_file(self, path: str) -> List[MLQEMData]:
        result = []
        with open(path, "r") as file:
            json_data = json.load(file)
            for entry in json_data:
                result.append(ExpValData.deserialize(entry))
        return result
