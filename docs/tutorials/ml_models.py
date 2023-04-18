import json, os
import glob, pickle

from tqdm.notebook import tqdm

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

from qiskit import QuantumCircuit
from qiskit.circuit.library import U3Gate, CZGate, PhaseGate, CXGate
from mbd_utils import construct_random_clifford, cal_z_exp, calc_imbalance, cal_all_z_exp, construct_mbl_circuit, generate_disorder, modify_and_add_noise_to_model
from gnn import ExpValCircuitGraphModel, ExpValCircuitGraphModel_2, ExpValCircuitGraphModel_3
from mlp import MLP1, MLP2, MLP3, encode_data
from collections import defaultdict
from sklearn.linear_model import LinearRegression

backend = FakeLima()
properties = get_backend_properties_v1(backend)

## Local
backend_ideal = QasmSimulator() # Noiseless
backend_noisy = AerSimulator.from_backend(FakeLima()) # Noisy

run_config_ideal = {'shots': 10000, 'backend': backend_ideal, 'name': 'ideal'}
run_config_noisy = {'shots': 10000, 'backend': backend_noisy, 'name': 'noisy'}


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


def train_gnn(train_paths, test_paths, save_path, num_epochs):
    BATCH_SIZE = 32

    fix_random_seed(0)

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

    fix_random_seed(0)
    model = ExpValCircuitGraphModel(
        num_node_features=22,
        hidden_channels=15
    )
    criterion = torch.nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer,
                                  'min',
                                  factor=0.1,
                                  patience=15,
                                  verbose=True,
                                  min_lr=0.00001)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(len(train_loader) * BATCH_SIZE, len(val_loader) * BATCH_SIZE)

    fix_random_seed(0)

    train_losses = []
    val_losses = []

    progress = tqdm(range(num_epochs), desc='Model training', leave=True)
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

    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.yscale('log')

    plt.style.use({'figure.facecolor': 'white'})
    plt.legend()
    plt.show()

    model_path = save_path+'/gnn1.pth'

    print("saved:", model_path)
    torch.save(model.state_dict(), model_path)

    import pickle
    to_save = {'train_losses': train_losses, 'val_losses': val_losses}
    with open('.'+model_path.split('.')[1]+'.pk', 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##########################################################################################
    model.eval()
    distances = []

    num_spins = 4
    even_qubits = np.linspace(0, num_spins, int(num_spins / 2), endpoint=False)
    odd_qubits = np.linspace(1, num_spins + 1, int(num_spins / 2), endpoint=False)

    for i, data in enumerate(val_loader):
        out = model(data.noisy_0, data.observable, data.circuit_depth, data.x, data.edge_index, data.batch)

        for ideal, noisy, ngm_mitigated in zip(
                data.y.tolist(),
                data.noisy_0.tolist(),
                out.tolist()
        ):
            imbalance_ideal = calc_imbalance(ideal, even_qubits, odd_qubits)[0]
            imbalance_noisy = calc_imbalance(noisy, even_qubits, odd_qubits)[0]
            imbalance_mitigated = calc_imbalance([ngm_mitigated], even_qubits, odd_qubits)[0]
            for q in range(4):
                ideal_q = ideal[0][q]
                noisy_q = noisy[0][q]
                ngm_mitigated_q = ngm_mitigated[q]
                distances.append({
                    f"ideal_{q}": ideal_q,
                    f"noisy_{q}": noisy_q,
                    f"ngm_mitigated_{q}": ngm_mitigated_q,
                    f"dist_noisy_{q}": np.abs(ideal_q - noisy_q),
                    f"dist_mitigated_{q}": np.abs(ideal_q - ngm_mitigated_q),
                    f"dist_sq_noisy_{q}": np.square(ideal_q - noisy_q),
                    f"dist_sq_mitigated_{q}": np.square(ideal_q - ngm_mitigated_q),
                    "imb_ideal": imbalance_ideal,
                    "imb_noisy": imbalance_noisy,
                    "imb_ngm": imbalance_mitigated,
                    "imb_diff": imbalance_ideal - imbalance_mitigated
                })

    plt.style.use({'figure.facecolor': 'white'})

    df = pd.DataFrame(distances)

    for q in range(4):
        print(f'RMSE_noisy_{q}:', np.sqrt(df[f"dist_sq_noisy_{q}"].mean()))
        print(f'RMSE_mitigated_{q}:', np.sqrt(df[f"dist_sq_mitigated_{q}"].mean()))

    print(f'RMSE_noisy:', np.sqrt(np.mean([df[f"dist_sq_noisy_{q}"].mean() for q in range(4)])))
    print(f'RMSE_mitigated:', np.sqrt(np.mean([df[f"dist_sq_mitigated_{q}"].mean() for q in range(4)])))

    sns.boxplot(data=df[
        ["dist_noisy_0", "dist_mitigated_0", "dist_noisy_1", "dist_mitigated_1", "dist_noisy_2", "dist_mitigated_2",
         "dist_noisy_3", "dist_mitigated_3"]], orient="h", showfliers=False)
    plt.title("Dist to ideal exp value")
    plt.show()

    sns.histplot([df['ideal_0'], df['noisy_0'], df["ngm_mitigated_0"]], kde=True, bins=40)
    plt.title("Exp values distribution")
    plt.show()


def train_mlp():
    raise NotImplementedError


