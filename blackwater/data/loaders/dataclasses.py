"""Loaders for dataclasses."""
from typing import Optional, List, Union

from torch_geometric import transforms as pyg_transforms
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform

from blackwater.data.dataclasses import ExpValData
from blackwater.data.io.io import ExpValDataReader


class ExpValDataLoader(Dataset):
    """ExpValDataLoader."""

    def __init__(
        self,
        path: Union[str, List[str]],
        transforms: Optional[List[BaseTransform]] = None,
    ):
        """Loader for ExpValData entries.

        Args:
            path: path or list of paths to json files with ExpValData data.
            transforms: list of transformations to perform on graphs
        """
        super().__init__()

        transforms = transforms or [pyg_transforms.AddSelfLoops()]
        paths = path if isinstance(path, list) else [path]
        self.paths = paths

        self.entries = []

        reader = ExpValDataReader()

        for file_path in paths:
            data: List[ExpValData] = reader.read_from_file(file_path)  # type: ignore[assignment]
            for entry in data:
                entry = entry.to_pyg()
                for transform in transforms:
                    entry = transform(entry)
                self.entries.append(entry)

    def len(self):
        """Number of entries."""
        return len(self.entries)

    def get(self, idx):
        """Get specific entry."""
        return self.entries[idx]
