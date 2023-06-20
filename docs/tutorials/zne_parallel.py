from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer.noise import depolarizing_error, NoiseModel
from qiskit.quantum_info import SparsePauliOp

from qiskit.primitives import BackendEstimator

from zne import zne, ZNEStrategy
from zne.noise_amplification import *
from zne.extrapolation import *

import numpy as np
import matplotlib.pyplot as plt

import json, os, pickle, random
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
from qiskit.providers.fake_provider import FakeMontreal, FakeLima, FakeGuadalupe, FakeJakarta

from blackwater.data.utils import (
    # generate_random_pauli_sum_op,
    # create_estimator_meas_data,
    # circuit_to_graph_data_json,
    get_backend_properties_v1,
    # encode_pauli_sum_op,
    # create_meas_data_from_estimators
)

from mlp import MLP1, MLP2, MLP3, encode_data

from mbd_utils import cal_z_exp, generate_disorder, construct_mbl_circuit, calc_imbalance, modify_and_add_noise_to_model

import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing
from multiprocessing import Pool
from functools import partial
from noise_utils import AddNoise, RemoveReadoutErrors
from qiskit.circuit import Barrier

# ############################################################
# backend = FakeGuadalupe()
# properties = get_backend_properties_v1(backend)
#
# ## Local
# backend_ideal = QasmSimulator()  # Noiseless
# backend_noisy = AerSimulator.from_backend(backend)  # Noisy
# ###########################################################

############################################################
# backend = FakeLima()
# properties = get_backend_properties_v1(backend)
#
# # Local, coherent noise
# backend_ideal = QasmSimulator() # Noiseless
# backend_noisy_coherent, noise_model = AddNoise(backend=backend).add_coherent_noise(seed=0, theta=np.pi * 0.04, uniform=False, add_depolarization=True)
############################################################

############################################################
backend = FakeLima()
properties = get_backend_properties_v1(backend)

# Local, coherent noise
backend_ideal = QasmSimulator() # Noiseless
backend_noisy_no_readout = RemoveReadoutErrors().remove_readout_errors()[0]
############################################################


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


def load_circuits(data_dir, f_ext='.json'):
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


def load_circuits_with_obs(data_dir, f_ext='.json'):
    circuits = []
    ideal_exp_vals = []
    noisy_exp_vals = []
    meas_basis = []
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
                meas_basis.append(entry['meas_basis'])
    return circuits, ideal_exp_vals, noisy_exp_vals, meas_basis


def get_measurement_qubits(qc, num_measured_qubit):
    measurement_qubits = []
    for measurement in range(num_measured_qubit - 1, -1, -1):
        measurement_qubits.append(qc.data[-1 - measurement][1][0].index)
    return measurement_qubits


def get_all_circuit_meas_mapping(circs):
    out = []
    for circ in circs:
        mapping = get_measurement_qubits(circ, 4)
        mapping = mapping[::-1]
        for i in range(5):
            if i not in mapping:
                mapping = mapping + [i]
        out.append(mapping)
    return out


OB_LIST = np.array(['ZIIII', 'IZIII', 'IIZII', 'IIIZI', 'IIIIZ'][::-1])


def get_all_circuit_ob_list(mappings):
    out = []
    for m in mappings:
        ob_list = OB_LIST[np.array(m)][:-1].tolist()
        out.append(ob_list)
    return out


def get_zne_expval_parallel_single_z(
        circs,
        extrapolator,
        backend,
        noise_factors=(1, 3),
        amplifier=LocalFoldingAmplifier(gates_to_fold=2),
        shots: int = 10000,
) -> float:
    ZNEEstimator = zne(BackendEstimator)
    estimator = ZNEEstimator(backend=backend)

    zne_strategy = ZNEStrategy(
        noise_factors=noise_factors,
        noise_amplifier=amplifier,
        extrapolator=extrapolator
    )

    mappings = get_all_circuit_meas_mapping(circs)
    ob_list_all = get_all_circuit_ob_list(mappings)

    return zne_strategy, estimator, ob_list_all, shots, circs


def get_zne_expval_parallel(
        circs,
        extrapolator,
        backend,
        noise_factors=(1, 3),
        amplifier=LocalFoldingAmplifier(gates_to_fold=2),
        shots: int = 10000,
) -> float:
    ZNEEstimator = zne(BackendEstimator)
    zne_estimator = ZNEEstimator(backend=backend)

    zne_strategy = ZNEStrategy(
        noise_factors=noise_factors,
        noise_amplifier=amplifier,
        extrapolator=extrapolator
    )

    return zne_strategy, zne_estimator, shots, circs


def form_all_qubit_observable(observable, measurement_qubits, total_num_qubits):
    """Input observable in non-endian, output observable in endian"""
    assert len(observable) == len(measurement_qubits)
    converted_obs = list('I' * total_num_qubits)
    for qubit, basis in zip(measurement_qubits, list(observable)):
        converted_obs[qubit] = basis
    return ''.join(converted_obs)[::-1]


def remove_until_barrier(qc, obs):
    circuit = qc.copy()
    circuit.remove_final_measurements()
    data = list(circuit.data)

    if (set(obs) != {'Z'}) and (set(obs) != {'Z', 'I'}):
        data.reverse()
        for ind, instruction in enumerate(data):
            if isinstance(instruction[0], Barrier):
                break
        data = data[ind:]
        data.reverse()

    new_circuit = circuit.copy()
    new_circuit.data = data

    return new_circuit


############################ Single Z ##########################################################
#######################################################################################################################
# DATA_FOLDER = './data/haoran_mbd/random_circuits/val/'
# SAVE_PATH = './zne_mitigated/random_circuits.pk'
DATA_FOLDER = './data/ising_init_from_qasm_no_readout/val_extra/'
DEGREE = 1
SAVE_PATH = f'zne_mitigated/ising_init_from_qasm_no_readout_extra_degree{DEGREE}.pk'
BACKEND = backend_noisy_no_readout

test_circuits, test_ideal_exp_vals, test_noisy_exp_vals = load_circuits(DATA_FOLDER, '.pk')
print(len(test_circuits))
test_noisy_exp_vals = [x[0] for x in test_noisy_exp_vals]

extrapolator = PolynomialExtrapolator(degree=DEGREE)
zne_strategy, estimator, ob_list_all, shots, circs = get_zne_expval_parallel_single_z(test_circuits, extrapolator,
                                                                                      BACKEND)

def process_circ_ob_list(args):
    i, circ, ob_list = args
    ob_list = list(map(SparsePauliOp, ob_list))
    job = estimator.run([circ] * 4, ob_list, shots=shots, zne_strategy=zne_strategy)
    values = job.result().values
    return i, values


if __name__ == '__main__':
    ###############################################################################
    mitigated = np.zeros((len(circs), 4))
    iterable = [(i, circ, ob_list) for i, (circ, ob_list) in enumerate(zip(circs, ob_list_all))]
    iterable = tqdm(iterable, total=len(circs), desc="Processing", ncols=80)
    with Pool() as pool:
        results = pool.map(process_circ_ob_list, iterable)

    for i, values in results:
        mitigated[i] = values

    mitigated *= -1
    print(mitigated)

    with open(SAVE_PATH, 'wb') as file:
        pickle.dump(mitigated, file)
    ###############################################################################

    ###############################################################################
    # with open(SAVE_PATH, 'wb') as file:
    #     mitigated = pickle.load(file)
    ###############################################################################

    ###############################################################################
    fix_random_seed(0)
    distances = []

    num_spins = 4
    even_qubits = np.linspace(0, num_spins, int(num_spins / 2), endpoint=False)
    odd_qubits = np.linspace(1, num_spins + 1, int(num_spins / 2), endpoint=False)

    extrapolator = PolynomialExtrapolator(degree=DEGREE)

    sl = slice(0, 100000)
    for miti, ideal, noisy in tqdm(zip(mitigated[sl], test_ideal_exp_vals[sl], test_noisy_exp_vals[sl]),
                                   total=len(test_circuits[sl])):

        imbalance_ideal = calc_imbalance([ideal], even_qubits, odd_qubits)[0]
        imbalance_noisy = calc_imbalance([noisy], even_qubits, odd_qubits)[0]
        imbalance_mitigated = calc_imbalance([miti], even_qubits, odd_qubits)[0]
        for q in range(4):
            ideal_q = ideal[q]
            noisy_q = noisy[q]
            miti_q = miti[q]
            distances.append({
                f"ideal_{q}": ideal_q,
                f"noisy_{q}": noisy_q,
                f"ngm_mitigated_{q}": miti_q,
                f"dist_noisy_{q}": np.abs(ideal_q - noisy_q),
                f"dist_mitigated_{q}": np.abs(ideal_q - miti_q),
                f"dist_sq_noisy_{q}": np.square(ideal_q - noisy_q),
                f"dist_sq_mitigated_{q}": np.square(ideal_q - miti_q),
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

########################################################################################################################


################################## Arbitrary Obs ##########################################
############################################################################################
# DATA_FOLDER = './data/ising_init_from_qasm_tomo/'
# DEGREE = 1
# SHOTS = 10000
# SAVE_PATH = f'zne_mitigated/ising_init_from_qasm_tomo_degree{DEGREE}.pk'
# BACKEND = backend_noisy
#
# test_circuits, test_ideal_exp_vals, test_noisy_exp_vals, obs = load_circuits_with_obs(DATA_FOLDER, '.pk')
# print(len(test_circuits))
# test_noisy_exp_vals = [x[0] for x in test_noisy_exp_vals]
#
# extrapolator = PolynomialExtrapolator(degree=DEGREE)
# zne_strategy, zne_estimator, _, circs = get_zne_expval_parallel(test_circuits, extrapolator, BACKEND)
#
#
# def process_circ_ob_list(args):
#     i, circ, ob = args
#     padded_obs = form_all_qubit_observable(ob, get_measurement_qubits(circ, 6), BACKEND.configuration().num_qubits)
#     circ_no_meas_circ = remove_until_barrier(circ, ob)
#     job = zne_estimator.run(circ_no_meas_circ, SparsePauliOp(padded_obs), shots=SHOTS, zne_strategy=zne_strategy)
#     values = job.result().values
#     return i, values
#
#
# if __name__ == '__main__':
#     ###############################################################################
#     mitigated = np.zeros((len(circs), 1))
#     iterable = [(i, circ, ob) for i, (circ, ob) in enumerate(zip(circs, obs))]
#     iterable = tqdm(iterable, total=len(circs), desc="Processing", ncols=80)
#     with Pool() as pool:
#         results = pool.map(process_circ_ob_list, iterable)
#
#     for i, values in results:
#         mitigated[i] = values
#
#     with open(SAVE_PATH, 'wb') as file:
#         pickle.dump(mitigated, file)
#     ###############################################################################
#
