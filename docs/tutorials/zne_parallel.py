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
from qiskit.providers.fake_provider import FakeMontreal, FakeLima

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
from noise_utils import AddNoise

############################################################
# backend = FakeLima()
# properties = get_backend_properties_v1(backend)
#
# ## Local
# backend_ideal = QasmSimulator()  # Noiseless
# backend_noisy = AerSimulator.from_backend(FakeLima())  # Noisy
#
# run_config_ideal = {'shots': 10000, 'backend': backend_ideal, 'name': 'ideal'}
# run_config_noisy = {'shots': 10000, 'backend': backend_noisy, 'name': 'noisy'}
############################################################

############################################################
backend = FakeLima()
properties = get_backend_properties_v1(backend)

# Local, coherent noise
backend_ideal = QasmSimulator() # Noiseless
backend_noisy_coherent, noise_model = AddNoise(backend=backend).add_coherent_noise(seed=0, theta=np.pi * 0.04, uniform=False, add_depolarization=True)

run_config_ideal = {'shots': 10000, 'backend': backend_ideal, 'name': 'ideal'}
run_config_noisy_coherent = {'shots': 10000, 'backend': backend_noisy_coherent, 'name': 'noisy_coherent'}
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


def get_measurement_qubits(qc, num_qubit):
    measurement_qubits = []
    for measurement in range(num_qubit - 1, -1, -1):
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
        noise_factors=(1, 3, 5, 7),
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


########################################################################################################################
# DATA_FOLDER = './data/haoran_mbd/random_circuits/val/'
# SAVE_PATH = './zne_mitigated/random_circuits.pk'
DATA_FOLDER = './data/ising_init_from_qasm_coherent/val_extra/'
DEGREE = 2
SAVE_PATH = f'zne_mitigated/ising_init_from_qasm_coherent_extra_degree{DEGREE}.pk'
BACKEND = backend_noisy_coherent

test_circuits, test_ideal_exp_vals, test_noisy_exp_vals = load_circuits(DATA_FOLDER, '.pk')
print(len(test_circuits))
test_noisy_exp_vals = [x[0] for x in test_noisy_exp_vals]

extrapolator = PolynomialExtrapolator(degree=DEGREE)
zne_strategy, estimator, ob_list_all, shots, circs = get_zne_expval_parallel_single_z(test_circuits, extrapolator,
                                                                                      BACKEND)
########################################################################################################################


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
