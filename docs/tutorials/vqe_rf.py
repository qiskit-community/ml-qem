import os
from pathlib import Path
from tqdm import tqdm
import pickle

from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.providers.fake_provider import FakeLimaV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimator, Estimator

from tqdm import tqdm

from blackwater.data.utils import get_backend_properties_v1, encode_pauli_sum_op
from blackwater.exception import BlackwaterException

from qiskit import QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.providers import JobV1 as Job, Options, BackendV2, Backend, BackendV1
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import FakeLimaV2, FakeLima, ConfigurableFakeBackend
from qiskit.opflow import I, X, Z, Y
from qiskit.algorithms.minimum_eigensolvers import VQE, VQEResult
from qiskit.algorithms.optimizers import SLSQP, SPSA, COBYLA, ADAM
from qiskit.circuit.library import TwoLocal

from blackwater.data.utils import generate_random_pauli_sum_op, get_backend_properties_v1
from blackwater.library.learning.estimator import learning, EmptyProcessor, TorchLearningModelProcessor, \
    ScikitLearningModelProcessor
from qiskit_aer import AerSimulator, QasmSimulator
import itertools
import numpy as np
from mbd_utils import cal_all_z_exp

from mlp import encode_data, MLP1
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import random

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from qiskit.result import marginal_counts
from qiskit import execute
from qiskit_aer import QasmSimulator
from qiskit.quantum_info import Operator

qasm_sim = QasmSimulator()

backend = FakeLima()
properties = get_backend_properties_v1(backend)

backend_ideal = QasmSimulator()  # Noiseless
backend_noisy = AerSimulator.from_backend(backend)  # Noisy

run_config_ideal = {'shots': 10000, 'backend': backend_ideal, 'name': 'ideal'}
run_config_noisy = {'shots': 10000, 'backend': backend_noisy, 'name': 'noisy'}

NUM_QUBITS = 2


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



def get_all_z_exp_wo_shot_noise(circuit, marginal_over=None):
    circuit_copy = circuit.copy()
    circuit_copy.remove_final_measurements()
    circuit_copy.save_density_matrix()

    def int_to_bin(n, num_bits=4):
        if n < 2 ** num_bits:
            binary_str = bin(n)[2:]
            return binary_str.zfill(num_bits)
        else:
            raise ValueError

    circuit_copy = transpile(circuit_copy, backend=backend_noisy, optimization_level=3)
    job = qasm_sim.run(circuit_copy)
    # job = execute(circuit_copy, QasmSimulator(), backend_options={'method': 'statevector'})
    probs = np.real(np.diag(job.result().results[0].data.density_matrix))
    probs = {int_to_bin(i, num_bits=NUM_QUBITS): p for i, p in enumerate(probs)}

    if marginal_over:
        probs = marginal_counts(probs, indices=marginal_over)

    exp_val = 0
    for key, prob in probs.items():
        num_ones = key.count('1')
        exp_val += (-1) ** num_ones * prob

    return exp_val



def load_circuits(data_dir, f_ext='.json', specific_file=None):
    circuits = []
    trans_circuits = []
    ideal_exp_vals = []
    noisy_exp_vals = []
    meas_basis = []
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f_ext)] if specific_file is None else [specific_file]
    for data_file in tqdm(data_files, leave=True):
        if f_ext == '.json':
            for entry in json.load(open(data_file, 'r')):
                circuits.append(QuantumCircuit.from_qasm_str(entry['circuit']))
                ideal_exp_vals.append(entry['ideal_exp_value'])
                noisy_exp_vals.append(entry['noisy_exp_values'])
        elif f_ext == '.pk':
            for entry in pickle.load(open(data_file, 'rb')):
                circuits.append(entry['circuit'])
                trans_circuits.append(entry['trans_circuit'])
                ideal_exp_vals.append(entry['ideal_exp_value'])
                noisy_exp_vals.append(entry['noisy_exp_values'])
                meas_basis.append(entry['meas_basis'])
    return trans_circuits, ideal_exp_vals, noisy_exp_vals, meas_basis




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    circuits, ideal_exp_vals, noisy_exp_vals, meas_bases = load_circuits('./data/vqe/', '.pk', specific_file='./data/vqe/two_local_2q_3reps_oplev0_rycz.pk')
    print(len(circuits))

    sep = 2999
    train_circuits, train_ideal_exp_vals, train_noisy_exp_vals, train_meas_bases = circuits[:sep], ideal_exp_vals[:sep], \
        noisy_exp_vals[:sep], meas_bases[:sep]
    test_circuits, test_ideal_exp_vals, test_noisy_exp_vals, test_meas_bases = circuits[sep:], ideal_exp_vals[sep:], \
        noisy_exp_vals[sep:], meas_bases[sep:]
    print(len(train_circuits))
    #################################################################################

    train_observables = [encode_pauli_sum_op(SparsePauliOp(basis))[0] for basis in train_meas_bases]
    test_observables = [encode_pauli_sum_op(SparsePauliOp(basis))[0] for basis in test_meas_bases]
    X_train, y_train = encode_data(train_circuits, properties, train_ideal_exp_vals, train_noisy_exp_vals, num_qubits=1,
                                   meas_bases=train_observables)
    X_test, y_test = encode_data(test_circuits, properties, test_ideal_exp_vals, test_noisy_exp_vals, num_qubits=1,
                                 meas_bases=test_observables)

    print(len(X_test[0]), 54 + 1 + len(test_observables[0]))

    #################################################################################
    BATCH_SIZE = 32
    fix_random_seed(0)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 1000, shuffle=False)

    #################################################################################
    print(train_circuits[0].count_ops())
    # train_circuits[0].decompose().draw('mpl', fold=-1, idle_wires=False).show()

    #################################################################################
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)

    #################################################################################
    fix_random_seed(0)
    distances = []
    num_spins = 1
    for batch_X, batch_y in test_loader:
        out = rfr.predict(batch_X)

        for ideal, noisy, ngm_mitigated in zip(
                batch_y.tolist(),
                batch_X[:, 54].tolist(),
                out.tolist()
        ):
            for q in range(num_spins):
                ideal_q = ideal[q]
                noisy_q = noisy
                ngm_mitigated_q = ngm_mitigated
                distances.append({
                    "num_train_samples": sep,
                    f"ideal_{q}": ideal_q,
                    f"noisy_{q}": noisy_q,
                    f"ngm_mitigated_{q}": ngm_mitigated_q,
                    f"dist_noisy_{q}": np.abs(ideal_q - noisy_q),
                    f"dist_mitigated_{q}": np.abs(ideal_q - ngm_mitigated_q),
                    f"dist_sq_noisy_{q}": np.square(ideal_q - noisy_q),
                    f"dist_sq_mitigated_{q}": np.square(ideal_q - ngm_mitigated_q),
                })

    plt.style.use({'figure.facecolor': 'white'})

    df = pd.DataFrame(distances)

    for q in range(num_spins):
        print(f'RMSE_noisy_{q}:', np.sqrt(df[f"dist_sq_noisy_{q}"].mean()))
        print(f'RMSE_mitigated_{q}:', np.sqrt(df[f"dist_sq_mitigated_{q}"].mean()))

    print(f'RMSE_noisy:', np.sqrt(np.mean([df[f"dist_sq_noisy_{q}"].mean() for q in range(num_spins)])))
    print(f'RMSE_mitigated:', np.sqrt(np.mean([df[f"dist_sq_mitigated_{q}"].mean() for q in range(num_spins)])))

    # sns.boxplot(data=df[["dist_noisy_0", "dist_mitigated_0"]], orient="h", showfliers=False)
    # plt.title("Dist to ideal exp value")
    # plt.show()

    # sns.histplot([df['ideal_0'], df['noisy_0'], df["ngm_mitigated_0"]], kde=True, bins=40)
    # plt.title("Exp values distribution")
    # # plt.xlim([-0.2, 0.2])
    # plt.show()

    #################################################################################
    fix_random_seed(0)

    processor = ScikitLearningModelProcessor(
        model=rfr,
        backend=backend_noisy
    )

    ##################################################################################
    str2opflow = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    coefficient = [0.1, 0.3, 0.7] #[0.2, 0.4, 0.4]
    operator_components = ['XX', 'ZZ', 'ZI']
    # num_ops_total, num_ops_from_train = 2, 2
    # coefficient = np.random.normal(0, 1, num_ops_total)
    # operator_components = np.random.choice(train_meas_bases, size=num_ops_from_train).tolist() + np.random.choice(test_meas_bases, size=num_ops_total-num_ops_from_train).tolist()
    print(operator_components)

    operator_components_opflow = []
    for op_component in operator_components:
        op_f = 1
        for op_str in list(op_component):
            # op_f = str2opflow[op_str] ^ op_f
            op_f = op_f ^ str2opflow[op_str]
        operator_components_opflow.append(op_f)

    operator = np.dot(coefficient, operator_components_opflow)
    operator = SparsePauliOp.from_operator(operator)
    print(operator)
    #########################################################################################

    ##########################################################################################
    # fix_random_seed(0)
    def callback_func(lst, values, params):
        print(f'Values: {values}', f'Params: {params}')
        lst.append(values)
    optimizer = COBYLA(maxiter=100)
    ansatz = TwoLocal(num_qubits=NUM_QUBITS, rotation_blocks="ry", entanglement_blocks="cz", reps=3)
    init_pt = np.random.uniform(-5, 5, ansatz.num_parameters)

    learning_estimator = learning(BackendEstimator, processor=processor, backend=FakeLima(), skip_transpile=True)
    estimator_mitigated = learning_estimator(backend=FakeLima())
    history_mitigated = []
    vqe = VQE(estimator=estimator_mitigated, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
              callback=lambda a, params, values, d: callback_func(history_mitigated, values, params))
    result_mitigated = vqe.compute_minimum_eigenvalue(operator)

    ##########################################################################################
    # fix_random_seed(0)
    estimator_ideal = Estimator()
    history_ideal = []
    vqe = VQE(estimator=estimator_ideal, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
              callback=lambda a, params, values, d: callback_func(history_ideal, values, params))
    result_ideal = vqe.compute_minimum_eigenvalue(operator)

    ##########################################################################################
    # fix_random_seed(0)
    estimator_noisy = BackendEstimator(backend=FakeLima())
    history_noisy = []
    vqe = VQE(estimator=estimator_noisy, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
              callback=lambda a, params, values, d: callback_func(history_noisy, values, params))
    result_noisy = vqe.compute_minimum_eigenvalue(operator)

    ##########################################################################################
    print('#' * 50)
    print("Noisy", result_noisy.optimal_value)
    print("Mitigated", result_mitigated.optimal_value)
    print("Ideal", result_ideal.optimal_value)
    print("Diagonalization", min(np.real_if_close(np.linalg.eig(Operator(operator))[0])))
    sns.lineplot(history_ideal, label='ideal')
    sns.lineplot(history_mitigated, label='mitigated')
    sns.lineplot(history_noisy, label='noisy')
    plt.legend()
    plt.show()