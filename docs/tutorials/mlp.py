import qiskit.circuit.random
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

from qiskit import QuantumCircuit


class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP1, self).__init__()
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
        super(MLP2, self).__init__()
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



class MLP3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size, hidden_size//3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_size//3, output_size)

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
        x4 = self.fc3(x3)
        x4 = self.relu3(x4)
        x4 = self.dropout3(x4)
        x_out = self.fc4(x4)
        return x_out



def fix_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'random seed fixed to {seed}')


def count_gates_by_rotation_angle(circuit, bin_size):
    angles = []
    for instr, qargs, cargs in circuit.data:
        if instr.name in ['rx', 'ry', 'rz'] and len(qargs) == 1:
            angles += [float(instr.params[0])]
    bin_edges = np.arange(-2 * np.pi, 2 * np.pi + bin_size, bin_size)
    counts, _ = np.histogram(angles, bins=bin_edges)
    bin_labels = [f"{left:.2f} to {right:.2f}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
    angle_bins = {label: count for label, count in zip(bin_labels, counts)}
    return list(angle_bins.values())


def recursive_dict_loop(my_dict, parent_key=None, out=None, target_key1=None, target_key2=None):
    if out is None: out = []

    for key, val in my_dict.items():
        if isinstance(val, dict):
            recursive_dict_loop(val, key, out, target_key1, target_key2)
        else:
            if parent_key and target_key1 in str(parent_key) and key == target_key2:
                out += [val]
    return out or 0.



def encode_data(circuits, properties, ideal_exp_vals, noisy_exp_vals, num_qubits, meas_bases=None):
    if isinstance(noisy_exp_vals[0], list) and len(noisy_exp_vals[0]) == 1:
        noisy_exp_vals = [x[0] for x in noisy_exp_vals]

    gates_set = sorted(properties['gates_set'])     # must sort!

    if meas_bases is None:
        meas_bases = [[]]

    vec = [np.mean(recursive_dict_loop(properties, out=[], target_key1='cx', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='id', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='sx', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='x', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='rz', target_key2='gate_error'))]
    vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='', target_key2='readout_error'))]
    vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='', target_key2='t1'))]
    vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='', target_key2='t2'))]
    vec = torch.tensor(vec) * 100  # put it in the same order of magnitude as the expectation values

    bin_size = 0.1 * np.pi
    num_angle_bins = int(np.ceil(4 * np.pi / bin_size))

    X = torch.zeros([len(circuits), len(vec) + len(gates_set) + num_angle_bins + num_qubits + len(meas_bases[0])])

    vec_slice = slice(0, len(vec))
    gate_counts_slice = slice(len(vec), len(vec)+len(gates_set))
    angle_bins_slice = slice(len(vec)+len(gates_set), len(vec)+len(gates_set)+num_angle_bins)
    exp_val_slice = slice(len(vec)+len(gates_set)+num_angle_bins, len(vec)+len(gates_set)+num_angle_bins+num_qubits)
    meas_basis_slice = slice(len(vec)+len(gates_set)+num_angle_bins+num_qubits, len(X[0]))

    X[:, vec_slice] = vec[None, :]

    for i, circ in enumerate(circuits):
        gate_counts_all = circ.count_ops()
        X[i, gate_counts_slice] = torch.tensor(
            [gate_counts_all.get(key, 0) for key in gates_set]
        ) * 0.01  # put it in the same order of magnitude as the expectation values

    for i, circ in enumerate(circuits):
        gate_counts = count_gates_by_rotation_angle(circ, bin_size)
        X[i, angle_bins_slice] = torch.tensor(gate_counts) * 0.01  # put it in the same order of magnitude as the expectation values

        if num_qubits > 1: assert len(noisy_exp_vals[i]) == num_qubits
        elif num_qubits == 1: assert isinstance(noisy_exp_vals[i], float)

        X[i, exp_val_slice] = torch.tensor(noisy_exp_vals[i])

    if meas_bases != [[]]:
        assert len(meas_bases) == len(circuits)
        for i, basis in enumerate(meas_bases):
            X[i, meas_basis_slice] = torch.tensor(basis)

    y = torch.tensor(ideal_exp_vals, dtype=torch.float32)

    return X, y



# def encode_data_old(circuits, properties, ideal_exp_vals, noisy_exp_vals, num_qubits):
#     gates_set = sorted(properties['gates_set'])     # must sort!
#
#     vec = [np.mean(recursive_dict_loop(properties, out=[], target_key1='cx', target_key2='gate_error'))]
#     vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='id', target_key2='gate_error'))]
#     vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='sx', target_key2='gate_error'))]
#     vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='x', target_key2='gate_error'))]
#     vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='rz', target_key2='gate_error'))]
#     vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='', target_key2='readout_error'))]
#     vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='', target_key2='t1'))]
#     vec += [np.mean(recursive_dict_loop(properties, out=[], target_key1='', target_key2='t2'))]
#     vec = torch.tensor(vec) * 100  # put it in the same order of magnitude as the expectation values
#     bin_size = 0.1 * np.pi
#     num_angle_bins = int(np.ceil(4 * np.pi / bin_size))
#
#     X = torch.zeros([len(circuits), len(vec) + len(gates_set) + num_angle_bins + num_qubits])
#
#     X[:, :len(vec)] = vec[None, :]
#
#     for i, circ in enumerate(circuits):
#         gate_counts_all = circ.count_ops()
#         X[i, len(vec):len(vec) + len(gates_set)] = torch.tensor(
#             [gate_counts_all.get(key, 0) for key in gates_set]
#         ) * 0.01  # put it in the same order of magnitude as the expectation values
#
#     for i, circ in enumerate(circuits):
#         gate_counts = count_gates_by_rotation_angle(circ, bin_size)
#         X[i, len(vec) + len(gates_set): -num_qubits] = torch.tensor(gate_counts) * 0.01  # put it in the same order of magnitude as the expectation values
#
#         if num_qubits > 1: assert len(noisy_exp_vals[i]) == num_qubits
#         elif num_qubits == 1: assert isinstance(noisy_exp_vals[i], float)
#
#         X[i, -num_qubits:] = torch.tensor(noisy_exp_vals[i])
#
#     y = torch.tensor(ideal_exp_vals, dtype=torch.float32)
#
#     return X, y



if __name__ == '__main__':
    from qiskit.providers.fake_provider import FakeMontreal, FakeLima
    from blackwater.data.utils import get_backend_properties_v1
    from qiskit.quantum_info import SparsePauliOp
    from blackwater.data.utils import encode_pauli_sum_op
    backend = FakeLima()
    print(list({g.gate for g in backend.properties().gates}))
    properties = get_backend_properties_v1(backend)

    circuit_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncreg meas[4];\nrz(pi/2) q[1];\nsx q[1];\nrz(-pi/2) q[1];\ncx q[2],q[1];\nsx q[1];\nrz(1.7278759594743862) q[1];\nsx q[1];\nrz(-3.3042128505379065) q[1];\nsx q[2];\nrz(2.9845130209103035) q[2];\nsx q[2];\nrz(-0.9186429592174754) q[2];\nrz(-pi/4) q[3];\nsx q[3];\nrz(-pi/2) q[3];\nx q[4];\ncx q[4],q[3];\nrz(1.3504439735577742) q[3];\nsx q[3];\nrz(-2.343957397662077) q[3];\nsx q[3];\nrz(-1.4156195211105356) q[3];\ncx q[1],q[3];\nsx q[1];\nrz(-1.725836788316796) q[1];\nsx q[1];\nrz(-3.1162625188573028) q[1];\ncx q[2],q[1];\nsx q[1];\nrz(1.7278759594743862) q[1];\nsx q[1];\nrz(-3.3042128505379065) q[1];\nsx q[2];\nrz(2.9845130209103035) q[2];\nsx q[2];\nrz(-0.9186429592174754) q[2];\nrz(1.906049550783556) q[3];\nsx q[3];\nrz(-0.4486568037754637) q[3];\nsx q[3];\nrz(-1.201830429096015) q[3];\nsx q[4];\nrz(pi/20) q[4];\nsx q[4];\nrz(2.0916100803511672) q[4];\ncx q[4],q[3];\nrz(-1.3504439735577738) q[3];\nsx q[3];\nrz(-0.7976352559277142) q[3];\nsx q[3];\nrz(1.7259731324792575) q[3];\ncx q[1],q[3];\nsx q[1];\nrz(-1.725836788316796) q[1];\nsx q[1];\nrz(-3.1162625188573028) q[1];\ncx q[2],q[1];\nsx q[1];\nrz(1.7278759594743862) q[1];\nsx q[1];\nrz(-3.3042128505379065) q[1];\nsx q[2];\nrz(2.9845130209103035) q[2];\nsx q[2];\nrz(-0.9186429592174754) q[2];\nrz(1.906049550783556) q[3];\nsx q[3];\nrz(-0.4486568037754637) q[3];\nsx q[3];\nrz(-1.201830429096015) q[3];\nsx q[4];\nrz(-2.9845130209103035) q[4];\nsx q[4];\nrz(2.0916100803511655) q[4];\ncx q[4],q[3];\nrz(-1.3504439735577738) q[3];\nsx q[3];\nrz(-0.7976352559277142) q[3];\nsx q[3];\nrz(1.7259731324792575) q[3];\ncx q[1],q[3];\nsx q[1];\nrz(-1.725836788316796) q[1];\nsx q[1];\nrz(-3.1162625188573028) q[1];\ncx q[2],q[1];\nsx q[1];\nrz(1.7278759594743862) q[1];\nsx q[1];\nrz(-3.3042128505379065) q[1];\nsx q[2];\nrz(2.9845130209103035) q[2];\nsx q[2];\nrz(-0.9186429592174754) q[2];\nrz(1.906049550783556) q[3];\nsx q[3];\nrz(-0.4486568037754637) q[3];\nsx q[3];\nrz(-1.201830429096015) q[3];\nsx q[4];\nrz(-2.9845130209103035) q[4];\nsx q[4];\nrz(2.0916100803511655) q[4];\ncx q[4],q[3];\nrz(-1.3504439735577738) q[3];\nsx q[3];\nrz(-0.7976352559277142) q[3];\nsx q[3];\nrz(1.7259731324792575) q[3];\ncx q[1],q[3];\nsx q[1];\nrz(-1.725836788316796) q[1];\nsx q[1];\nrz(-3.1162625188573028) q[1];\ncx q[2],q[1];\nsx q[1];\nrz(1.7278759594743862) q[1];\nsx q[1];\nrz(-3.3042128505379065) q[1];\nsx q[2];\nrz(2.9845130209103035) q[2];\nsx q[2];\nrz(-0.9186429592174754) q[2];\nrz(1.906049550783556) q[3];\nsx q[3];\nrz(-0.4486568037754637) q[3];\nsx q[3];\nrz(-1.201830429096015) q[3];\nsx q[4];\nrz(-2.9845130209103035) q[4];\nsx q[4];\nrz(2.0916100803511655) q[4];\ncx q[4],q[3];\nrz(-1.3504439735577738) q[3];\nsx q[3];\nrz(-0.7976352559277142) q[3];\nsx q[3];\nrz(1.7259731324792575) q[3];\ncx q[1],q[3];\nsx q[1];\nrz(-1.725836788316796) q[1];\nsx q[1];\nrz(-3.1162625188573028) q[1];\ncx q[2],q[1];\nsx q[1];\nrz(1.7278759594743862) q[1];\nsx q[1];\nrz(-3.3042128505379065) q[1];\nsx q[2];\nrz(2.9845130209103035) q[2];\nsx q[2];\nrz(-0.9186429592174754) q[2];\nrz(1.906049550783556) q[3];\nsx q[3];\nrz(-0.4486568037754637) q[3];\nsx q[3];\nrz(-1.201830429096015) q[3];\nsx q[4];\nrz(-2.9845130209103035) q[4];\nsx q[4];\nrz(2.0916100803511655) q[4];\ncx q[4],q[3];\nrz(-1.3504439735577738) q[3];\nsx q[3];\nrz(-0.7976352559277142) q[3];\nsx q[3];\nrz(1.7259731324792575) q[3];\ncx q[1],q[3];\nsx q[1];\nrz(-1.725836788316796) q[1];\nsx q[1];\nrz(-3.1162625188573028) q[1];\ncx q[2],q[1];\nsx q[1];\nrz(1.7278759594743862) q[1];\nsx q[1];\nrz(-3.3042128505379065) q[1];\nsx q[2];\nrz(2.9845130209103035) q[2];\nsx q[2];\nrz(-0.9186429592174754) q[2];\nrz(1.906049550783556) q[3];\nsx q[3];\nrz(-0.4486568037754637) q[3];\nsx q[3];\nrz(-1.201830429096015) q[3];\nsx q[4];\nrz(-2.9845130209103035) q[4];\nsx q[4];\nrz(1.036297431763694) q[4];\ncx q[4],q[3];\nrz(-1.3504439735577738) q[3];\nsx q[3];\nrz(-0.7976352559277142) q[3];\nsx q[3];\nrz(1.7259731324792575) q[3];\ncx q[1],q[3];\nsx q[1];\nrz(-1.725836788316796) q[1];\nsx q[1];\nrz(-3.1162625188573028) q[1];\ncx q[2],q[1];\nsx q[1];\nrz(1.7278759594743862) q[1];\nsx q[1];\nrz(-pi) q[1];\nsx q[2];\nrz(2.9845130209103035) q[2];\nsx q[2];\nrz(-0.9186429592174754) q[2];\nsx q[3];\nrz(1.7278759594743862) q[3];\nsx q[3];\nrz(-4.349014256811735) q[3];\nsx q[4];\nrz(-1.4928778811476757) q[4];\nsx q[4];\nrz(-0.13653026392406042) q[4];\ncx q[3],q[4];\nsx q[3];\nrz(1.7278759594743862) q[3];\nsx q[3];\ncx q[1],q[3];\nsx q[1];\nrz(2.9845130209103035) q[1];\nsx q[1];\nrz(-0.16065255166893166) q[1];\nsx q[3];\nrz(1.7278759594743862) q[3];\nsx q[3];\nrz(-1.2074216032219418) q[3];\nsx q[4];\nrz(1.7278759594743862) q[4];\nsx q[4];\nrz(-2.091610080351167) q[4];\nbarrier q[2],q[1],q[3],q[4];\nmeasure q[2] -> meas[0];\nmeasure q[1] -> meas[1];\nmeasure q[3] -> meas[2];\nmeasure q[4] -> meas[3];\n'
    circuits = [QuantumCircuit.from_qasm_str(circuit_qasm)]
    # circuits = [qiskit.circuit.random.random_circuit(4, 10, 2, measure=True, seed=0)]
    ideal_exp_vals = [1, -1, 1, -1]
    noisy_exp_vals = [[0.9, -0.92, 0.89, -0.94]]

    for _ in range(1):
        # X1, y1 = encode_data_old(circuits, properties, ideal_exp_vals, noisy_exp_vals, 4)
        # print(X1[0, :])
        meas_bases = encode_pauli_sum_op(SparsePauliOp('XYZI'))
        print(meas_bases)
        X2, y2 = encode_data(circuits, properties, ideal_exp_vals, noisy_exp_vals, 4, meas_bases=meas_bases)
        print(X2[0, :])
        # assert (X1 == X2).all()
        # assert (y1 == y2).all()