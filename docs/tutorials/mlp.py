import torch
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
        X[i, -num_qubits:] = torch.tensor(noisy_exp_vals[i])

    y = torch.tensor(ideal_exp_vals)

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


if __name__ == '__main__':
    with open('./data/mbd_datasets2/theta_0.05pi/circuits.pk', 'rb') as infile:
        loaded = pickle.load(infile)

    train_circuits = loaded['train_circuits']
    test_circuits = loaded['test_circuits']
    train_ideal_exp_vals = loaded['train_ideal_exp_vals']
    train_noisy_exp_vals = loaded['train_noisy_exp_vals']
    test_ideal_exp_vals = loaded['test_ideal_exp_vals']
    test_noisy_exp_vals = loaded['test_noisy_exp_vals']

    print('data loaded..')

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

    N_EPOCHS = 50

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

    print('training done... ')

    model.eval()
    distances = []

    num_spins = 4
    even_qubits = np.linspace(0, num_spins, int(num_spins / 2), endpoint=False)
    odd_qubits = np.linspace(1, num_spins + 1, int(num_spins / 2), endpoint=False)

    for batch_X, batch_y in test_loader:
        out = model(batch_X)

        for ideal, noisy, ngm_mitigated in zip(
                batch_y.tolist(),
                batch_X[:, -4:].tolist(),
                out.tolist()
        ):
            imbalance_ideal = calc_imbalance([ideal], even_qubits, odd_qubits)[0]
            imbalance_noisy = calc_imbalance([noisy], even_qubits, odd_qubits)[0]
            imbalance_mitigated = calc_imbalance([ngm_mitigated], even_qubits, odd_qubits)[0]
            ideal = np.mean(ideal)
            noisy = np.mean(noisy)
            ngm_mitigated = np.mean(ngm_mitigated)
            distances.append({
                "ideal": ideal,
                "noisy": noisy,
                "ngm_mitigated": ngm_mitigated,
                "dist_noisy": np.abs(ideal - noisy),
                "dist_ngm": np.abs(ideal - ngm_mitigated),
                "imb_ideal": imbalance_ideal,
                "imb_noisy": imbalance_noisy,
                "imb_ngm": imbalance_mitigated,
                "imb_diff": imbalance_ideal - imbalance_mitigated
            })

    plt.style.use({'figure.facecolor': 'white'})

    df = pd.DataFrame(distances)

    sns.histplot([df['imb_ideal'], df['imb_noisy'], df["imb_ngm"]], kde=True, bins=40)
    plt.title("Exp values distribution")
    plt.show()

    sns.histplot(data=df["imb_diff"], kde=True, bins=40)
    plt.title("Dist to ideal exp value")
    plt.show()

    print('test on disorder...')

    ## Now we need to average over many disorders
    num_disorders = 10

    num_spins = 4  # Number of spins. Must be even.
    W = 0.8 * np.pi  # Disorder strength up to np.pi
    theta = 0.05 * np.pi  # Interaction strength up to np.pi
    max_steps = 10

    even_qubits = np.linspace(0, num_spins, int(num_spins / 2), endpoint=False)
    odd_qubits = np.linspace(1, num_spins + 1, int(num_spins / 2), endpoint=False)

    # For each disorder realization, make a new disorder
    # and compute the charge imbalance using the same physics parameters as before
    imbalance_all_ideal = []
    imbalance_all_noisy = []
    imbalance_all_mitigated = []

    for disorder_realization in tqdm(range(num_disorders)):
        disorders = generate_disorder(num_spins, W)
        # print(disorders)

        qc_list = []
        k = np.random.choice(list(range(490)))
        for steps in range(max_steps):
            qc_list.append(train_circuits[k + steps * 500])
            # qc_list.append(construct_mbl_circuit(num_spins, disorders, theta, steps))

        transpiled_qc_list = transpile(qc_list, backend_noisy, optimization_level=3)
        job_ideal = execute(qc_list, **run_config_ideal)
        job_noisy = execute(transpiled_qc_list, **run_config_noisy)

        exp_Z_ideal = []
        exp_Z_noisy = []
        exp_Z_mitigated = []

        for i in range(len(qc_list)):
            counts_ideal = job_ideal.result().get_counts()[i]
            counts_noisy = job_noisy.result().get_counts()[i]

            ideal_exp_val = cal_z_exp(counts_ideal)
            noisy_exp_val = cal_z_exp(counts_noisy)

            exp_Z_ideal.append(list(ideal_exp_val))  # Single-Z expectation value of each qubit
            exp_Z_noisy.append(list(noisy_exp_val))  # Single-Z expectation value of each qubit

            # print(entry.batch)
            X, _ = encode_data([transpiled_qc_list[i]], properties, ideal_exp_val, noisy_exp_val, num_qubits=4)
            mitigated_exp_val = model(X).tolist()[0]

            exp_Z_mitigated.append(mitigated_exp_val)

        imbalance_ideal = calc_imbalance(exp_Z_ideal, even_qubits, odd_qubits)
        imbalance_noisy = calc_imbalance(exp_Z_noisy, even_qubits, odd_qubits)
        imbalance_mitigated = calc_imbalance(exp_Z_mitigated, even_qubits, odd_qubits)

        imbalance_all_ideal.append(imbalance_ideal)
        imbalance_all_noisy.append(imbalance_noisy)
        imbalance_all_mitigated.append(imbalance_mitigated)

    # Average imbalance
    imbalance_ideal_average = np.mean(imbalance_all_ideal, axis=0)
    imbalance_noisy_average = np.mean(imbalance_all_noisy, axis=0)
    imbalance_mitigated_average = np.mean(imbalance_all_mitigated, axis=0)

    ## Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    for disorder_realization in range(num_disorders):
        ax1.plot(imbalance_all_ideal[disorder_realization], color=(0.0, 0.0, 1.0, 0.1))
        ax1.plot(imbalance_all_noisy[disorder_realization], color=(1.0, 0.0, 0.0, 0.1))
        ax1.plot(imbalance_all_mitigated[disorder_realization], color=(0.0, 1.0, 0.0, 0.1))

    ax1.plot(imbalance_ideal_average, color='blue', label="ideal")
    ax1.plot(imbalance_noisy_average, color='red', label="noisy")
    ax1.plot(imbalance_mitigated_average, color='green', label="mitigated")

    ax1.axvline(x=10, color='gray', label='training data availability')

    ax1.set_xlabel('Floquet steps')
    ax1.set_ylabel('Imbalance')
    ax1.legend()

    xmin, xmax = ax1.get_xlim()
    max_x = max_steps + 10
    ax1.set_xlim([0, max_x - 1])
    ax2.set_xlim([0, max_x - 1])
    # ax1.set_ylim([0.4, 1.03])
    ax1.set_xticks(np.arange(0, max_x, 4))
    x2 = np.linspace(xmin, xmax, 50)
    ax2.plot(x2, -np.ones(50))  # Create a dummy plot
    ax2.set_xticks(np.arange(0, max_x, 4))
    ax2.set_xticklabels(2 * np.arange(0, max_x, 4))
    ax2.set_xlabel(r"2q gate depth")
    # ax1.grid(None)
    ax2.grid(None)

    plt.style.use({'figure.facecolor': 'white'})
    plt.show()
