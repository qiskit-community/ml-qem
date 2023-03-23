import torch, random
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

import numpy as np

import json, os, pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import CXGate, RXGate, IGate, ZGate
from qiskit.providers.fake_provider import FakeMontreal, FakeLima

from blackwater.data.utils import (
    generate_random_pauli_sum_op,
    create_estimator_meas_data,
    circuit_to_graph_data_json,
    get_backend_properties_v1,
    encode_pauli_sum_op,
    create_meas_data_from_estimators
)

from mbd_utils import cal_z_exp, generate_disorder, construct_mbl_circuit, calc_imbalance

import matplotlib.pyplot as plt
import seaborn as sns


backend = FakeLima()
properties = get_backend_properties_v1(backend)

## Local
backend_ideal = QasmSimulator()  # Noiseless
backend_noisy = AerSimulator.from_backend(FakeLima())  # Noisy

run_config_ideal = {'shots': 10000, 'backend': backend_ideal, 'name': 'ideal'}
run_config_noisy = {'shots': 10000, 'backend': backend_noisy, 'name': 'noisy'}


def encode_data(circuits, properties, ideal_exp_vals, noisy_exp_vals, num_qubits):
    gates_set = properties['gates_set']

    def recursive_dict_loop(my_dict, parent_key=None, out=[], target_key1=None, target_key2=None):
        for key, val in my_dict.items():
            if isinstance(val, dict):
                recursive_dict_loop(val, key, out, target_key1, target_key2)
            else:
                if parent_key and target_key1 in str(parent_key) and key == target_key2:
                    out += [val]
        return out

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

    def count_gates_by_rotation_angle(circuit):
        angles = []
        for instr, qargs, cargs in circuit.data:
            if instr.name in ['rx', 'ry', 'rz'] and len(qargs) == 1:
                angles += [instr.params[0]]
        bin_edges = np.arange(-2 * np.pi, 2 * np.pi + bin_size, bin_size)
        counts, _ = np.histogram(angles, bins=bin_edges)
        bin_labels = [f"{left:.2f} to {right:.2f}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
        angle_bins = {label: count for label, count in zip(bin_labels, counts)}
        return list(angle_bins.values())

    for i, circ in enumerate(circuits):
        gate_counts = count_gates_by_rotation_angle(circ)
        X[i, len(vec) + len(gates_set): -num_qubits] = torch.tensor(
            gate_counts) * 0.01  # put it in the same order of magnitude as the expectation values
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


num_qubit = 4
W = 0.8 * np.pi
theta = 0.05 * np.pi
def construct_random_mbd_func(num_steps):
    disorders = generate_disorder(num_qubit, W)
    random_mbl = construct_mbl_circuit(num_qubit, disorders, theta, num_steps)
    circuit = transpile(random_mbl, backend=backend_noisy, optimization_level=3)
    return circuit


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    with open('../data/mbd_datasets2/theta_0.05pi/circuits.pk', 'rb') as infile:
        loaded = pickle.load(infile)

    train_circuits = loaded['train_circuits']
    test_circuits = loaded['test_circuits']
    train_ideal_exp_vals = loaded['train_ideal_exp_vals']
    train_noisy_exp_vals = loaded['train_noisy_exp_vals']
    test_ideal_exp_vals = loaded['test_ideal_exp_vals']
    test_noisy_exp_vals = loaded['test_noisy_exp_vals']

    # train_ideal_exp_vals = []
    # train_noisy_exp_vals = []
    # train_circuits = []
    # for depth in range(0, 10, 2):
    #     for i in tqdm(range(500)):
    #         circuit = construct_random_mbd_func(depth)
    #         train_circuits.append(circuit)
    #
    #         job_ideal = execute(circuit, **run_config_ideal)
    #         job_noisy = execute(circuit, **run_config_noisy)
    #
    #         counts_ideal = job_ideal.result().get_counts()
    #         counts_noisy = job_noisy.result().get_counts()
    #
    #         ideal_exp_val = cal_z_exp(counts_ideal)
    #         noisy_exp_val = cal_z_exp(counts_noisy)
    #
    #         train_ideal_exp_vals.append(ideal_exp_val)
    #         train_noisy_exp_vals.append(noisy_exp_val)
    #
    # to_save = {'train_circuits': train_circuits, 'train_ideal_exp_vals': train_ideal_exp_vals, 'train_noisy_exp_vals': train_noisy_exp_vals}
    # with open('./data/tmp_train_circuits.pk', 'wb') as out:
    #     pickle.dump(to_save, out)


    # test_ideal_exp_vals = []
    # test_noisy_exp_vals = []
    # test_circuits = []
    # for depth in range(0, 10, 2):
    #     for i in tqdm(range(100)):
    #         circuit = construct_random_mbd_func(depth)
    #         test_circuits.append(circuit)
    #
    #         job_ideal = execute(circuit, **run_config_ideal)
    #         job_noisy = execute(circuit, **run_config_noisy)
    #
    #         counts_ideal = job_ideal.result().get_counts()
    #         counts_noisy = job_noisy.result().get_counts()
    #
    #         ideal_exp_val = cal_z_exp(counts_ideal)
    #         noisy_exp_val = cal_z_exp(counts_noisy)
    #
    #         test_ideal_exp_vals.append(ideal_exp_val)
    #         test_noisy_exp_vals.append(noisy_exp_val)

    # to_save = {'test_circuits': test_circuits, 'test_ideal_exp_vals': test_ideal_exp_vals, 'test_noisy_exp_vals': test_noisy_exp_vals}
    # with open('./data/tmp_test_circuits.pk', 'wb') as out:
    #     pickle.dump(to_save, out)
    #
    # with open('./data/tmp_train_circuits.pk', 'rb') as in_f:
    #     loaded = pickle.load(in_f)
    # train_circuits = loaded['train_circuits']
    # train_ideal_exp_vals = loaded['train_ideal_exp_vals']
    # train_noisy_exp_vals = loaded['train_noisy_exp_vals']
    #
    # with open('./data/tmp_test_circuits.pk', 'rb') as in_f:
    #     loaded = pickle.load(in_f)
    # test_circuits = loaded['test_circuits']
    # test_ideal_exp_vals = loaded['test_ideal_exp_vals']
    # test_noisy_exp_vals = loaded['test_noisy_exp_vals']

    print('data loaded..')

    train_noisy_exp_vals = [x[0] for x in train_noisy_exp_vals]
    test_noisy_exp_vals = [x[0] for x in test_noisy_exp_vals]

    X_train, y_train = encode_data(train_circuits, properties, train_ideal_exp_vals, train_noisy_exp_vals, num_qubits=4)
    X_test, y_test = encode_data(test_circuits, properties, test_ideal_exp_vals, test_noisy_exp_vals, num_qubits=4)

    BATCH_SIZE = 32

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP1(
        input_size=58,
        output_size=4,
        hidden_size=128
    )


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(len(train_loader) * BATCH_SIZE, len(test_loader) * BATCH_SIZE)

    train_losses = []
    test_losses = []

    N_EPOCHS = 30

    print('start training ...')

    progress = tqdm(range(N_EPOCHS), desc='Model training', leave=True)
    for epoch in progress:
        train_loss = 0.0
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        test_loss = 0.0
        model.eval()
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

        if epoch >= 1:
            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))

            progress.set_description(f"{round(train_losses[-1], 5)}, {round(test_losses[-1], 5)}")
            progress.refresh()

    plt.plot(train_losses, label="train_loss")
    plt.plot(test_losses, label="test_loss")
    plt.yscale('log')

    plt.style.use({'figure.facecolor': 'white'})
    plt.legend()
    plt.show()

    print('training done... ')

    model.eval()
    distances = []


    # indices = random.sample(list(range(len(train_circuits))), 100)
    indices = list(range(10, 2500, 10))
    qc_list = [train_circuits[j] for j in indices]
    train_ideal_exp_vals_selected = [train_ideal_exp_vals[j] for j in indices]
    train_noisy_exp_vals_selected = [train_noisy_exp_vals[j] for j in indices]
    # test_ideal_exp_vals_selected = [test_ideal_exp_vals[j] for j in indices]
    # test_noisy_exp_vals_selected = [test_noisy_exp_vals[j] for j in indices]
    print('sampled...')
    # qc_list.append(construct_mbl_circuit(num_spins, disorders, theta, steps))

    # for i in range(len(qc_list)):
    #     X, y = encode_data([qc_list[i]], properties, train_ideal_exp_vals_selected[i], [train_noisy_exp_vals_selected[i]], num_qubits=4)
    #     out = model(X)

    for batch_X, batch_y in test_loader:
        out = model(batch_X)

        for ideal, noisy, ngm_mitigated in zip(
                batch_y.tolist(),
                batch_X[:, -4:].tolist(),
                out.tolist()
                # [y.tolist()],
                # [train_noisy_exp_vals_selected[i]],
                # out.tolist()
        ):
            ideal = np.mean(ideal)
            noisy = np.mean(noisy)
            ngm_mitigated = np.mean(ngm_mitigated)
            distances.append({
                "ideal": ideal,
                "noisy": noisy,
                "ngm_mitigated": ngm_mitigated,
                "dist_noisy": np.abs(ideal - noisy),
                "dist_ngm": np.abs(ideal - ngm_mitigated),
            })

    plt.style.use({'figure.facecolor': 'white'})

    df = pd.DataFrame(distances)

    sns.histplot([df['ideal'], df['noisy'], df["ngm_mitigated"]], kde=True, bins=100)
    plt.title("Exp values distribution")
    plt.show()

    print('test on disorder...')


    qc_list = []
    for steps in range(10):
        # qc_list.append(train_circuits[np.random.choice(list(range(190))) + steps * 500])
        disorders = generate_disorder(4, W)
        qc_list.append(construct_mbl_circuit(4, disorders, theta, steps))

    transpiled_qc_list = transpile(qc_list, backend_noisy, optimization_level=3)
    job_ideal = execute(transpiled_qc_list, **run_config_ideal)
    job_noisy = execute(transpiled_qc_list, **run_config_noisy)
    print('job executed...')

    exp_Z_ideal = []
    exp_Z_noisy = []
    exp_Z_mitigated = []
    exp_Z_mitigated2 = []
    exp_Z_mitigated3 = []
    exp_Z_mitigated4 = []
    to_compare_noisy = []
    to_compare_ideal = []


    for i in tqdm(range(len(qc_list))):
        counts_ideal = job_ideal.result().get_counts()[i]
        counts_noisy = job_noisy.result().get_counts()[i]

        ideal_exp_val = cal_z_exp(counts_ideal)
        noisy_exp_val = cal_z_exp(counts_noisy)

        X, _ = encode_data([transpiled_qc_list[i]], properties, ideal_exp_val, [noisy_exp_val], num_qubits=4)
        mitigated_exp_val = model(X).tolist()[0]

        # X2, _ = encode_data([transpiled_qc_list[i]], properties, train_ideal_exp_vals_selected[i], [train_noisy_exp_vals_selected[i]], num_qubits=4)
        # mitigated_exp_val2 = model(X2).tolist()[0]
        #
        # X3, _ = encode_data([transpiled_qc_list[i]], properties, ideal_exp_val, [train_noisy_exp_vals_selected[i]], num_qubits=4)
        # mitigated_exp_val3 = model(X3).tolist()[0]
        #
        # X4, _ = encode_data([transpiled_qc_list[i]], properties, train_ideal_exp_vals_selected[i], [noisy_exp_val], num_qubits=4)
        # mitigated_exp_val4 = model(X4).tolist()[0]

        # to_compare_noisy.append((noisy_exp_val, train_noisy_exp_vals[indices[i]], f + np.array(train_noisy_exp_vals[indices[i]])))
        # to_compare_ideal.append((ideal_exp_val, train_ideal_exp_vals[indices[i]], f + np.array(train_ideal_exp_vals[indices[i]])))

        exp_Z_ideal.append(np.mean(ideal_exp_val))  # Single-Z expectation value of each qubit
        exp_Z_noisy.append(np.mean(noisy_exp_val))  # Single-Z expectation value of each qubit
        exp_Z_mitigated.append(np.mean(mitigated_exp_val))
        # exp_Z_mitigated2.append(np.mean(mitigated_exp_val2))
        # exp_Z_mitigated3.append(np.mean(mitigated_exp_val3))
        # exp_Z_mitigated4.append(np.mean(mitigated_exp_val4))

    sns.histplot([exp_Z_ideal, exp_Z_noisy, exp_Z_mitigated], kde=True, bins=80)
    plt.show()

    # sns.histplot([exp_Z_ideal, np.mean(train_ideal_exp_vals_selected, 1), exp_Z_mitigated, exp_Z_mitigated2], kde=True, bins=40)
    # plt.show()
    #
    # sns.histplot([exp_Z_mitigated3, exp_Z_mitigated4], kde=True, bins=40)
    # plt.show()

    # sns.histplot([exp_Z_ideal, np.mean(train_ideal_exp_vals_selected, 1)])
    # plt.show()

    # sns.histplot([exp_Z_noisy, np.mean(train_noisy_exp_vals_selected, 1)])
    # plt.show()
    #
    # print(to_compare_noisy)
    # print(to_compare_ideal)