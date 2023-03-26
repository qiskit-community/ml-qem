import torch, random
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

import numpy as np
import json, os, pickle
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def count_gates_by_rotation_angle(circuit):
    angles = []
    for instr, qargs, cargs in circuit.data:
        if instr.name in ['rx', 'ry', 'rz'] and len(qargs) == 1:
            angles += [float(instr.params[0])]
    bin_edges = np.arange(-2 * np.pi, 2 * np.pi + bin_size, bin_size)
    counts, _ = np.histogram(angles, bins=bin_edges)
    bin_labels = [f"{left:.2f} to {right:.2f}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
    angle_bins = {label: count for label, count in zip(bin_labels, counts)}
    return list(angle_bins.values())


def recursive_dict_loop(my_dict, parent_key=None, out=[], target_key1=None, target_key2=None):
    for key, val in my_dict.items():
        if isinstance(val, dict):
            recursive_dict_loop(val, key, out, target_key1, target_key2)
        else:
            if parent_key and target_key1 in str(parent_key) and key == target_key2:
                out += [val]
    return out


def encode_data(circuits, properties, ideal_exp_vals, noisy_exp_vals, num_qubits):
    gates_set = properties['gates_set']

    vec = [np.mean(recursive_dict_loop(properties, target_key1='cx', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, target_key1='id', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, target_key1='sx', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, target_key1='x', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, target_key1='rz', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, target_key1='', target_key2='readout_error'))]
    vec += [np.mean(recursive_dict_loop(properties, target_key1='', target_key2='t1'))]
    vec += [np.mean(recursive_dict_loop(properties, target_key1='', target_key2='t2'))]
    vec = torch.tensor(vec) * 100  # put it in the same order of magnitude as the expectation values

    bin_size = 0.1 * np.pi
    num_angle_bins = int(np.ceil(4 * np.pi / bin_size))

    X = torch.zeros([len(circuits), len(vec) + len(gates_set) + num_angle_bins + num_qubits])

    X[:, :len(vec)] = vec[None, :]

    for i, circ in enumerate(circuits):
        gate_counts = circ.count_ops()
        X[i, len(vec):len(vec) + len(gates_set)] = torch.tensor(
            [gate_counts.get(key, 0) for key in gates_set]
        ) * 0.01  # put it in the same order of magnitude as the expectation values

    for i, circ in enumerate(circuits):
        gate_counts = count_gates_by_rotation_angle(circ)
        X[i, len(vec) + len(gates_set): -num_qubits] = torch.tensor(gate_counts) * 0.01  # put it in the same order of magnitude as the expectation values

        if num_qubits > 1: assert len(noisy_exp_vals[i]) == num_qubits
        elif num_qubits == 1: assert noisy_exp_vals[i].isnumeric()

        X[i, -num_qubits:] = torch.tensor(noisy_exp_vals[i])

    y = torch.tensor(ideal_exp_vals, dtype=torch.float32)

    return X, y


class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # First layer
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)

        # Second layer
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)

        # Skip connection
        x3 = x1 + x2

        # Output layer
        x_out = self.fc3(x3)
        return x_out
