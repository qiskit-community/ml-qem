import json
import glob

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit import execute
from qiskit.providers.fake_provider import FakeLima
from qiskit.primitives import Estimator
from qiskit.circuit.random import random_circuit

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import dropout

from torch_geometric.nn import GCNConv, global_mean_pool, Linear, ChebConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm.notebook import tqdm_notebook
import matplotlib.pyplot as plt
import seaborn as sns

from blackwater.data.loaders.exp_val import CircuitGraphExpValMitigationDataset
from blackwater.data.generators.exp_val import exp_value_generator
from blackwater.data.utils import generate_random_pauli_sum_op
from blackwater.library.ngem.estimator import ngem

from qiskit.quantum_info import random_clifford

import random
from qiskit.circuit.library import HGate, SdgGate
from qiskit.circuit import ClassicalRegister

from blackwater.data.utils import (
    generate_random_pauli_sum_op,
    create_estimator_meas_data,
    circuit_to_graph_data_json,
    get_backend_properties_v1,
    encode_pauli_sum_op,
    create_meas_data_from_estimators
)
from blackwater.data.generators.exp_val import ExpValueEntry
from blackwater.metrics.improvement_factor import improvement_factor, Trial, Problem

from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.providers.fake_provider import FakeMontreal, FakeLima

from torch_geometric.nn import (
    GCNConv,
    TransformerConv,
    GATv2Conv,
    global_mean_pool,
    Linear,
    ChebConv,
    SAGEConv,
    ASAPooling,
    dense_diff_pool,
    avg_pool_neighbor_x
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch


class ExpValCircuitGraphModel(torch.nn.Module):
    def __init__(
            self,
            num_node_features: int,
            hidden_channels: int
    ):
        super().__init__()

        self.transformer1 = TransformerConv(
            num_node_features, hidden_channels,
            heads=3,
            dropout=0.1
        )
        self.pooling1 = ASAPooling(hidden_channels * 3, 0.5)

        self.transformer2 = TransformerConv(
            hidden_channels * 3, hidden_channels,
            heads=2,
            dropout=0.1
        )
        self.pooling2 = ASAPooling(hidden_channels * 2, 0.5)

        self.body_seq = torch.nn.Sequential(
            Linear(hidden_channels * 2 + 5, hidden_channels),
            torch.nn.Dropout(0.2),
            Linear(hidden_channels, 4)
        )

    def forward(self,
                exp_value, observable,
                circuit_depth, nodes,
                edge_index, batch):
        graph = self.transformer1(nodes, edge_index)
        graph, edge_index, _, batch, _ = self.pooling1(
            graph, edge_index, batch=batch
        )

        graph = self.transformer2(graph, edge_index)
        graph, edge_index, _, batch, _ = self.pooling2(
            graph, edge_index, batch=batch
        )

        graph = global_mean_pool(graph, batch)

        merge = torch.cat((
            graph,
            torch.squeeze(exp_value, 1),
            circuit_depth
        ), dim=1)

        return self.body_seq(merge)
