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

from tqdm import tqdm
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

from mlp import MLP3


class ExpValCircuitGraphModel(torch.nn.Module):
    def __init__(
            self,
            num_node_features: int,
            hidden_channels: int,
            exp_value_size: int = 4,
            dropout: float = 0.2
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
            Linear(hidden_channels * 2 + 1 + exp_value_size, hidden_channels),
            torch.nn.Dropout(dropout),
            Linear(hidden_channels, exp_value_size)
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



class ExpValCircuitGraphModel_2(torch.nn.Module):
    def __init__(
            self,
            num_node_features: int,
            hidden_channels: int,
            exp_value_size: int = 4,
            dropout: float = 0.5
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
        self.body_seq = MLP2(
            input_size=hidden_channels * 2 + 1 + exp_value_size,
            hidden_size=hidden_channels,
            output_size=exp_value_size,
            dropout_rate=dropout
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




class ExpValCircuitGraphModel_3(torch.nn.Module):
    def __init__(
            self,
            num_node_features: int,
            hidden_channels: int,
            exp_value_size: int = 4,
            dropout: float = 0.3
    ):
        super().__init__()
        self.transformer1 = TransformerConv(
            num_node_features, hidden_channels,
            heads=5,
            dropout=0.1
        )
        self.pooling1 = ASAPooling(hidden_channels * 5, 0.5)
        self.transformer2 = TransformerConv(
            hidden_channels * 5, hidden_channels,
            heads=3,
            dropout=0.1
        )
        self.pooling2 = ASAPooling(hidden_channels * 3, 0.5)
        self.body_seq = MLP3(
            input_size=hidden_channels * 3 + 1 + exp_value_size,
            hidden_size=hidden_channels * 5,
            output_size=exp_value_size,
            dropout_rate=dropout
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




if __name__ == "__main__":
    train_paths = [
        './data/mbd_datasets2/theta_0.05pi/train/step_1.json',
    ]

    val_paths = [
        './data/mbd_datasets2/theta_0.05pi/val/step_1.json',
    ]

    BATCH_SIZE = 32

    train_loader = DataLoader(
        CircuitGraphExpValMitigationDataset(
            train_paths,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        CircuitGraphExpValMitigationDataset(
            val_paths,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    for data in train_loader:
        print(data.noisy_0.shape)
        break

    model = ExpValCircuitGraphModel_3(
        num_node_features=22,
        hidden_channels=15,
        exp_value_size=4,
    )
    criterion = torch.nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer,
                                  'min',
                                  factor=0.1,
                                  patience=15,
                                  verbose=True,
                                  min_lr=0.00001)

    min_valid_loss = np.inf

    train_losses = []
    val_losses = []

    N_EPOCHS = 100

    progress = tqdm(range(N_EPOCHS), desc='Model training', leave=True)
    for epoch in progress:
        train_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            out = model(
                data.noisy_0,
                data.observable,
                data.circuit_depth,
                data.x,
                data.edge_index,
                data.batch
            )
            loss = criterion(out, torch.squeeze(data.y, 1))

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        valid_loss = 0.0
        model.eval()
        for i, data in enumerate(val_loader):
            out = model(
                data.noisy_0,
                data.observable,
                data.circuit_depth,
                data.x,
                data.edge_index,
                data.batch)
            loss = criterion(out, torch.squeeze(data.y, 1))

            valid_loss += loss.item()

        scheduler.step(valid_loss)

        if epoch >= 1:
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(valid_loss / len(val_loader))

            progress.set_description(f"{round(train_losses[-1], 5)}, {round(val_losses[-1], 5)}")
            progress.refresh()
