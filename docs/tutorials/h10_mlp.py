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

from mlp import MLP1, MLP2, encode_data

from mbd_utils import cal_z_exp, generate_disorder, construct_mbl_circuit, calc_imbalance

import matplotlib.pyplot as plt
import seaborn as sns
import random

BATCH_SIZE = 32

backend = FakeLima()
properties = get_backend_properties_v1(backend)

## Local
backend_ideal = QasmSimulator()  # Noiseless
backend_noisy = AerSimulator.from_backend(FakeLima())  # Noisy

run_config_ideal = {'shots': 10000, 'backend': backend_ideal, 'name': 'ideal'}
run_config_noisy = {'shots': 10000, 'backend': backend_noisy, 'name': 'noisy'}
h10_test_data_path = './data/mbd_datasets2/theta_0.05pi/circuits.pk'

DATA_FILEPATH = "data_filepath"
MODEL_FILEPATH = "model_filepath"
LOAD_FUNCTION = "load_function"


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


def custom_loader(X, y, batch_size=BATCH_SIZE):
    assert len(X) == len(y)
    num_batches = int(np.ceil(len(X) / batch_size))
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    for i in range(num_batches):
        yield X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]


def load_circuits_d1_d2(data_dir, f_ext='.pk'):
    circuits = []
    ideal_exp_vals = []
    noisy_exp_vals = []
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f_ext)]
    for data_file in tqdm(data_files, leave=True):
        if f_ext == '.json':
            for entry in json.load(open(data_file, 'r')):
                circuits.append(QuantumCircuit.from_qasm_str(entry['circuit']))
                ideal_exp_vals.append(entry['ideal_exp_value'])
                noisy_exp_vals.append(entry['noisy_exp_values'])
        elif f_ext == '.pk':
            for entry in pickle.load(open(data_file, 'rb')):
                circuits.append(entry['circuit'])
                ideal_exp_vals.append(entry['ideal_exp_value'])
                noisy_exp_vals.append(entry['noisy_exp_values'])
    return circuits, ideal_exp_vals, noisy_exp_vals


def load_circuits_d3(data_file, **kwargs):
    with open(data_file, 'rb') as infile:
        loaded = pickle.load(infile)
    train_circuits = loaded['train_circuits']
    train_ideal_exp_vals = loaded['train_ideal_exp_vals']
    train_noisy_exp_vals = loaded['train_noisy_exp_vals']

    return train_circuits, train_ideal_exp_vals, train_noisy_exp_vals


def load_test_sets_d123():
    with open(h10_test_data_path, 'rb') as infile:
        loaded = pickle.load(infile)

    test_circuits = loaded['test_circuits']
    test_ideal_exp_vals = loaded['test_ideal_exp_vals']
    test_noisy_exp_vals = loaded['test_noisy_exp_vals']

    return test_circuits, test_ideal_exp_vals, test_noisy_exp_vals


PATH_TO_FOLDER = '/Users/haoranliao/GitHub/blackwater/docs/tutorials'

data_info = {
    "d1": {
        DATA_FILEPATH: PATH_TO_FOLDER + '/data/haoran_mbd/random_cliffords',
        LOAD_FUNCTION: load_circuits_d1_d2,
        "name": "random_cliffords",
        MODEL_FILEPATH: PATH_TO_FOLDER + "/model/haoran_mbd2/mlp_random_cliffords.pth"
    },
    "d2": {
        DATA_FILEPATH: PATH_TO_FOLDER + '/data/haoran_mbd/random_brickwork',
        LOAD_FUNCTION: load_circuits_d1_d2,
        "name": "random_brickwork",
        MODEL_FILEPATH: PATH_TO_FOLDER + "/model/haoran_mbd2/mlp_random_brickwork.pth"
    },
    "d3": {
        DATA_FILEPATH: PATH_TO_FOLDER + '/data/mbd_datasets2/theta_0.05pi/circuits.pk',
        LOAD_FUNCTION: load_circuits_d3,
        "name": 'mbd',
        MODEL_FILEPATH: PATH_TO_FOLDER + "/model/haoran_mbd2/mlp_mbd.pth"
    },
}


def load_test_loader_d123(test_circuits, test_ideal_exp_vals, test_noisy_exp_vals):
    test_noisy_exp_vals = [x[0] for x in test_noisy_exp_vals]
    X_test, y_test = encode_data(test_circuits, properties, test_ideal_exp_vals, test_noisy_exp_vals, num_qubits=4)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=np.random.seed(0))
    # test_loader = custom_loader(X_test, y_test)
    return test_loader


def train_model_d123(data_name):
    data_filepath = data_info[data_name][DATA_FILEPATH]
    retrieval_func = data_info[data_name][LOAD_FUNCTION]

    train_circuits, train_ideal_exp_vals, train_noisy_exp_vals = retrieval_func(data_filepath, f_ext='.pk')
    train_noisy_exp_vals = [x[0] for x in train_noisy_exp_vals]
    X_train, y_train = encode_data(train_circuits, properties, train_ideal_exp_vals, train_noisy_exp_vals, num_qubits=4)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=np.random.seed(0))
    test_circuits, test_ideal_exp_vals, test_noisy_exp_vals = load_test_sets_d123()
    test_loader = load_test_loader_d123(test_circuits, test_ideal_exp_vals, test_noisy_exp_vals)

    model = MLP1(
        input_size=58,
        output_size=4,
        hidden_size=128
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer,
                                  'min',
                                  factor=0.1,
                                  patience=15,
                                  verbose=True,
                                  min_lr=0.00001)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(len(train_loader) * BATCH_SIZE, len(test_loader) * BATCH_SIZE)

    train_losses = []
    test_losses = []

    N_EPOCHS = 50

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

        scheduler.step(test_loss)

        if epoch >= 1:
            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))

            progress.set_description(f"{round(train_losses[-1], 5)}, {round(test_losses[-1], 5)}")
            progress.refresh()

    return train_losses, test_losses, model


def plot_trained_model(train_losses, test_losses):
    plt.plot(train_losses, label="train_loss")
    plt.plot(test_losses, label="test_loss")
    plt.yscale('log')

    plt.style.use({'figure.facecolor': 'white'})
    plt.legend()
    plt.show()


def save_model(model, model_path):
    print('saved:', model_path)
    torch.save(model.state_dict(), model_path)


def load_model(model_path):
    model = MLP1(
        input_size=58,
        output_size=4,
        hidden_size=128
    )
    print('loaded:', model_path)
    model.load_state_dict(torch.load(model_path), strict=True)
    return model


def make_plot(model, name="default"):
    test_circuits, test_ideal_exp_vals, test_noisy_exp_vals = load_test_sets_d123()
    test_loader = load_test_loader_d123(test_circuits, test_ideal_exp_vals, test_noisy_exp_vals)
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
    # sns.boxplot(data=df[["dist_noisy", "dist_ngm"]], orient="h", showfliers=False)
    # plt.title("Dist to ideal exp value")
    # plt.show()

    # sns.histplot([df['ideal'], df['noisy'], df["ngm_mitigated"]], kde=True, bins=40)
    # plt.title("Exp values distribution")
    # plt.show()

    # sns.histplot([df['imb_ideal'], df['imb_noisy'], df["imb_ngm"]], kde=True, bins=40)
    # plt.title(f"{name}: Exp values distribution")
    # plt.show()

    sns.histplot(data=df["imb_diff"], kde=True, bins=40)
    plt.title(f"{name}: Dist to ideal exp value")
    plt.ylim([0, 600])
    plt.xlim([-0.2, 0.2])
    plt.show()


def run_and_save_model(data_name, extra=""):
    dinfo = data_info[data_name]
    train_losses, test_losses, model = train_model_d123(data_name)
    plot_trained_model(train_losses, test_losses)
    save_model(model, model_path=dinfo[MODEL_FILEPATH])
    make_plot(model, name=f"{data_name}_{extra}_trained model")


max_steps = 0
def run_floquet_steps(models):
    from collections import defaultdict
    num_disorders = 10
    num_spins = 4
    W = 0.8 * np.pi
    theta = 0.05 * np.pi
    even_qubits = np.linspace(0, num_spins, int(num_spins / 2), endpoint=False)
    odd_qubits = np.linspace(1, num_spins + 1, int(num_spins / 2), endpoint=False)

    imbalance_all_ideal = []
    imbalance_all_noisy = []
    imbalance_all_mitigated_dict = defaultdict(list)

    exp_Z_all_ideal = []
    exp_Z_all_noisy = []
    exp_Z_all_mitigated_dict = defaultdict(list)

    for disorder_realization in tqdm(range(num_disorders)):
        disorders = generate_disorder(num_spins, W, seed=disorder_realization)

        qc_list = []
        for steps in range(max_steps + 10):
            qc_list.append(construct_mbl_circuit(num_spins, disorders, theta, steps))

        transpiled_qc_list = transpile(qc_list, backend_noisy, optimization_level=3)
        job_ideal = execute(qc_list, **run_config_ideal)
        job_noisy = execute(transpiled_qc_list, **run_config_noisy)

        exp_Z_ideal = []
        exp_Z_noisy = []
        exp_Z_mitigated_dict = defaultdict(list)

        for i in range(len(qc_list)):
            counts_ideal = job_ideal.result().get_counts()[i]
            counts_noisy = job_noisy.result().get_counts()[i]

            ideal_exp_val = cal_z_exp(counts_ideal)
            noisy_exp_val = cal_z_exp(counts_noisy)

            exp_Z_ideal.append(list(ideal_exp_val))  # Single-Z expectation value of each qubit
            exp_Z_noisy.append(list(noisy_exp_val))  # Single-Z expectation value of each qubit

            X, _ = encode_data([transpiled_qc_list[i]], properties, ideal_exp_val, [noisy_exp_val], num_qubits=4)

            for j, model in enumerate(models):
                mitigated_exp_val = model(X).tolist()[0]
                exp_Z_mitigated_dict[j].append(mitigated_exp_val)

        imbalance_ideal = calc_imbalance(exp_Z_ideal, even_qubits, odd_qubits)
        imbalance_noisy = calc_imbalance(exp_Z_noisy, even_qubits, odd_qubits)
        imbalance_mitigated_dict = defaultdict(list)
        for k, exp_Z_mitigated in exp_Z_mitigated_dict.items():
            imbalance_mitigated_dict[k] = calc_imbalance(exp_Z_mitigated, even_qubits, odd_qubits)

        imbalance_all_ideal.append(imbalance_ideal)
        imbalance_all_noisy.append(imbalance_noisy)
        for k, imbalance_mitigated in imbalance_mitigated_dict.items():
            imbalance_all_mitigated_dict[k].append(imbalance_mitigated)

        exp_Z_all_ideal.append(exp_Z_ideal)
        exp_Z_all_noisy.append(exp_Z_noisy)
        for k, exp_Z_mitigated in exp_Z_mitigated_dict.items():
            exp_Z_all_mitigated_dict[k].append(exp_Z_mitigated)

    # Average imbalance
    imbalance_ideal_average = np.mean(imbalance_all_ideal, axis=0)
    imbalance_noisy_average = np.mean(imbalance_all_noisy, axis=0)
    imbalance_mitigated_average_dict = defaultdict()
    for ind, imbalance_all_mitigated in imbalance_all_mitigated_dict.items():
        imbalance_mitigated_average_dict[ind] = np.mean(imbalance_all_mitigated, axis=0)

    return imbalance_ideal_average, imbalance_noisy_average, imbalance_mitigated_average_dict


def plot_floquet_steps(imbalance_ideal_average, imbalance_noisy_average, imbalance_mitigated_average_dict):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(imbalance_ideal_average, color='blue', label="ideal")
    ax1.plot(imbalance_noisy_average, color='red', label="noisy")
    for i, imbalance_mitigated_average in imbalance_mitigated_average_dict.items():
        ax1.plot(imbalance_mitigated_average, color='green', label=f"MLP {i}", linestyle='solid')

    ax1.axvline(x=9, color='gray', label='training data availability')

    ax1.set_xlabel('Floquet steps')
    ax1.set_ylabel('Imbalance')
    ax1.legend()

    xmin, xmax = ax1.get_xlim()
    max_x = max_steps + 10
    ax1.set_xlim([0, max_x - 1])
    ax2.set_xlim([0, max_x - 1])
    ax1.set_ylim([0.4, 1.08])
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


def success_111():
    for data_name in ['d1']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"first_try")
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try")


def success_1211():
    for data_name in ['d1', 'd2']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"first_try")
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try")

def success_12311():
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"first_try")
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try")


def success_123121():
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")
    for data_name in ['d1', 'd2']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"first_try_{data_name}")
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try_{data_name}")


def success_1231231():
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"first_try_{data_name}")
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try_{data_name}")

def success_11i():
    for data_name in ['d1']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"first_try_{data_name}")

    model = MLP1(
        input_size=58,
        output_size=4,
        hidden_size=128
    )
    model.load_state_dict(torch.load(data_info['d1'][MODEL_FILEPATH]))
    make_plot(model, name=f"second_try_{data_name}")


def success_1i1():
    for data_name in ['d1']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")

    model = MLP1(
        input_size=58,
        output_size=4,
        hidden_size=128
    )
    model.load_state_dict(torch.load(data_info['d1'][MODEL_FILEPATH]))
    make_plot(model, name=f"first_try_{data_name}")

    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try_{data_name}")


def success_1l1():
    for data_name in ['d1']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")
    load_models()
    for data_name in ['d1']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try_{data_name}")


def success_123l123():
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")
    load_models()
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try_{data_name}")


def success_tllp():
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")

    for data_name in ['d1', 'd2', 'd3']:
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"first_try_{data_name}")

    models = []
    for data_name in ['d1', 'd2', 'd3']:
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        models.append(loaded_model)
        make_plot(loaded_model, name=f"second_try_{data_name}")

    fix_random_seed(0)
    plot_floquet_steps(*run_floquet_steps(models))


def success_tlpl():
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")

    models = []
    for data_name in ['d1', 'd2', 'd3']:
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        models.append(loaded_model)
        make_plot(loaded_model, name=f"first_try_{data_name}")

    fix_random_seed(0)
    plot_floquet_steps(*run_floquet_steps(models))

    for data_name in ['d1', 'd2', 'd3']:
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try_{data_name}")


def success_tlpll():
    for data_name in ['d1', 'd2', 'd3']:
        fix_random_seed(0)
        run_and_save_model(data_name, "")

    models = []
    for data_name in ['d1', 'd2', 'd3']:
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        models.append(loaded_model)

    fix_random_seed(0)
    plot_floquet_steps(*run_floquet_steps(models))

    for data_name in ['d1', 'd2', 'd3']:
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"second_try_{data_name}")

    for data_name in ['d1', 'd2', 'd3']:
        loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
        make_plot(loaded_model, name=f"third_try_{data_name}")


if __name__ == "__main__":
    success_tlpl()
    # for data_name in ['d1', 'd2', 'd3']:
    #     loaded_model = load_model(data_info[data_name][MODEL_FILEPATH])
    #     make_plot(loaded_model, name=f"second_try_{data_name}")

