import torch.nn


class NGSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, nodes, edge_index, edge_attr, unitary, batch):
        pass
