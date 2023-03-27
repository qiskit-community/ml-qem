"""ExpVal dataset."""
import json, pickle
from typing import Union, List, Optional, Dict, Any

from torch_geometric import transforms as pyg_transforms
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform

from blackwater.data.generators.exp_val import ExpValueEntry


# pylint: disable=abstract-method
class CircuitGraphExpValMitigationDataset(Dataset):
    """CircuitGraphMitigationDataset."""

    def __init__(
        self,
        path: Union[str, List[str]],
        transforms: Optional[List[BaseTransform]] = None,
        num_samples: int = None,
    ):
        """Dataset that contains graphs that preserve information
        about circuits and backends those circuits were executed on (gate errors, t1, t2, etc)
        and has noisy and ideal measurement results for those circuits

        Args:
            path: path or list of paths to files that contains json graph information
            transforms: list of transforms that will be applied to dataset
                default: adding self loops
        """
        super().__init__()

        transforms = transforms or [pyg_transforms.AddSelfLoops()]
        paths = path if isinstance(path, list) else [path]
        self.paths = paths

        self.entries = []


        for path_to_file in paths:
            if path_to_file.endswith('.json'):
                with open(
                    path_to_file, "r"
                ) as entries_file:
                    data: List[Dict[str, Any]] = json.load(entries_file)
                    if num_samples is not None:
                        data = data[:num_samples]

            elif path_to_file.endswith('.pk'):
                with open(
                    path_to_file, "rb"
                ) as entries_file:
                    data: List[Dict[str, Any]] = pickle.load(entries_file)
                    if num_samples is not None:
                        data = data[:num_samples]

            for entry in data:
                try:
                    entry.pop('circuit')
                except KeyError:
                    pass

                try:
                    entry.pop('metadata')
                except KeyError:
                    pass

                try:
                    data = ExpValueEntry.from_json(entry).to_pyg_data()

                    for transform in transforms:
                        data = transform(data)

                    self.entries.append(data)
                except KeyError:
                    pass

    def len(self):
        return len(self.entries)

    def get(self, idx):
        return self.entries[idx]
