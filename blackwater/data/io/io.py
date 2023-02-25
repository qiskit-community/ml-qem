import json
from typing import List

from blackwater.data.dataclasses import BlackwaterData, ExpValData
from blackwater.data.encoders.graph import GraphData


class Writer:
    def save_to_file(self, path: str, data: List[BlackwaterData]):
        raise NotImplementedError


class Reader:
    def read_from_file(self, path: str) -> List[BlackwaterData]:
        raise NotImplementedError


class ExpValDataWriter(Writer):
    def save_to_file(self, path: str, data: List[BlackwaterData]):
        with open(path, "w") as file:
            json.dump([
                entry.serialize()
                for entry in data
            ], file, indent=4)


class ExpValDataReader(Reader):
    def read_from_file(self, path: str) -> List[BlackwaterData]:
        result = []
        with open(path, "r") as file:
            json_data = json.load(file)
            for entry in json_data:
                result.append(ExpValData.deserialize(entry))
        return result
