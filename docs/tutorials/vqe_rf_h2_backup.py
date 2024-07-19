import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.opflow import I, X, Z
from qiskit.primitives import BackendEstimator, Estimator
from qiskit.providers.fake_provider import FakeLima
from qiskit.quantum_info import Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import marginal_counts
from qiskit_aer import AerSimulator
from qiskit_aer import QasmSimulator
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from zne import zne, ZNEStrategy
from zne.extrapolation import *
from zne.noise_amplification import *

from blackwater.data.utils import encode_pauli_sum_op
from blackwater.data.utils import get_backend_properties_v1
from blackwater.library.learning.estimator import learning, ScikitLearningModelProcessor, ZNEProcessor
from mlp import encode_data
from noise_utils import RemoveReadoutErrors

qasm_sim = QasmSimulator()

backend = FakeLima()
properties = get_backend_properties_v1(backend)

backend_ideal = QasmSimulator()  # Noiseless
backend_noisy = AerSimulator.from_backend(backend)  # Noisy

backend_noisy_wo_readout = RemoveReadoutErrors().remove_readout_errors()[0]

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
    circuits, ideal_exp_vals, noisy_exp_vals, meas_bases = load_circuits('./data/vqe/', '.pk', specific_file='./data/vqe/two_local_2q_3reps_oplev0_rycz_20240717.pk')
    print(len(circuits))

    combined = list(zip(circuits, ideal_exp_vals, noisy_exp_vals, meas_bases))
    random.seed(42)
    random.shuffle(combined)
    list1, list2, list3, list4 = zip(*combined)
    circuits, ideal_exp_vals, noisy_exp_vals, meas_bases = list(list1), list(list2), list(list3), list(list4)

    sep = 5000
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
    from sklearn.ensemble import RandomForestRegressor

    rfr = RandomForestRegressor(n_estimators=300)
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
    #################################################################################
    # repeat with no readout error
    circuits, ideal_exp_vals, noisy_exp_vals, meas_bases = load_circuits('./data/vqe/', '.pk', specific_file='./data/vqe/two_local_2q_3reps_oplev0_rycz_20240717_no_readout.pk')
    print(len(circuits))

    combined = list(zip(circuits, ideal_exp_vals, noisy_exp_vals, meas_bases))
    random.seed(42)
    random.shuffle(combined)
    list1, list2, list3, list4 = zip(*combined)
    circuits, ideal_exp_vals, noisy_exp_vals, meas_bases = list(list1), list(list2), list(list3), list(list4)

    sep = 5000
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
    from sklearn.ensemble import RandomForestRegressor

    rfr_wo_readout = RandomForestRegressor(n_estimators=300)
    rfr_wo_readout.fit(X_train, y_train)

    #################################################################################
    fix_random_seed(0)

    processor = ScikitLearningModelProcessor(
        model=rfr,
        backend=backend_noisy
    )
    processor_wo_readout = ScikitLearningModelProcessor(
        model=rfr_wo_readout,
        backend=backend_noisy_wo_readout
    )

    ZNEEstimator = zne(BackendEstimator)
    zne_estimator = ZNEEstimator(backend=backend_noisy)
    zne_w_readout_estimator = ZNEEstimator(backend=backend_noisy_wo_readout)
    zne_strategy = ZNEStrategy(
        noise_factors=(1, 3),
        noise_amplifier=LocalFoldingAmplifier(gates_to_fold=2),
        extrapolator=PolynomialExtrapolator(degree=1),
    )
    zne_processor = ZNEProcessor(
        zne_estimator=zne_estimator,
        zne_strategy=zne_strategy,
        backend=backend_noisy
    )
    zne_w_readout_processor = ZNEProcessor(
        zne_estimator=zne_w_readout_estimator,
        zne_strategy=zne_strategy,
        backend=backend_noisy_wo_readout
    )

    ##########################################################################################
    bond_operators = []

    with open("./h2-hamiltonian-qubit-params.txt", "r") as f:
        entries = f.read().split("\n\n")
        for entry in entries:
            if entry:
                length, fci, c1, c2, c3, c4, c5 = entry.strip().split("\n")
                length = float(length.split(" ")[0])
                fci = float(fci.split(" ")[-1])

                c1, c2, c3, c4, c5 = [
                    float(x.split(" ")[0]) for x in
                    [c1, c2, c3, c4, c5]
                ]

                operator_components_opflow = [I ^ I, X ^ X, Z ^ I, Z ^ Z, I ^ Z]
                coefficient = [c1, c2, c3, c4, c5]
                operator = np.dot(coefficient, operator_components_opflow)
                operator = SparsePauliOp.from_operator(operator)

                bond_operators.append((length, operator))
    ##########################################################################################
    bond_lengths = []
    mitigated = []
    mitigated_w_readout = []
    noisy = []
    zne = []
    zne_w_readout = []
    ideal = []
    diagonalization = []
    for bond_length, operator in bond_operators:
        # fix_random_seed(0)
        def callback_func(lst, values, params):
            print(f'Values: {values}', f'Params: {params}')
            lst.append(values)
        optimizer = COBYLA(maxiter=50)
        ansatz = TwoLocal(num_qubits=NUM_QUBITS, rotation_blocks="ry", entanglement_blocks="cz", reps=3)
        init_pt = np.ones(ansatz.num_parameters)

        ##########################################################################################
        learning_estimator = learning(BackendEstimator, processor=processor, backend=FakeLima(), skip_transpile=True)
        estimator_mitigated = learning_estimator(backend=FakeLima())
        history_mitigated = []
        vqe = VQE(estimator=estimator_mitigated, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
                  callback=lambda a, params, values, d: callback_func(history_mitigated, values, params))
        result_mitigated = vqe.compute_minimum_eigenvalue(operator, separate_observables=True)

        ##########################################################################################
        learning_w_readout_estimator = learning(BackendEstimator, processor=processor_wo_readout, backend=backend_noisy_wo_readout, skip_transpile=True)
        estimator_mitigated_w_readout = learning_estimator(backend=backend_noisy_wo_readout)
        history_mitigated_w_readout = []
        vqe = VQE(estimator=estimator_mitigated_w_readout, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
                  callback=lambda a, params, values, d: callback_func(history_mitigated_w_readout, values, params))
        result_mitigated_w_readout = vqe.compute_minimum_eigenvalue(operator, separate_observables=True)

        ##########################################################################################
        # fix_random_seed(0)
        estimator_ideal = Estimator()
        history_ideal = []
        vqe = VQE(estimator=estimator_ideal, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
                  callback=lambda a, params, values, d: callback_func(history_ideal, values, params))
        result_ideal = vqe.compute_minimum_eigenvalue(operator)

        ##########################################################################################
        # fix_random_seed(0)
        estimator_noisy = BackendEstimator(backend=backend_noisy)
        history_noisy = []
        vqe = VQE(estimator=estimator_noisy, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
                  callback=lambda a, params, values, d: callback_func(history_noisy, values, params))
        result_noisy = vqe.compute_minimum_eigenvalue(operator)

        ##########################################################################################
        # optimal_circuit_from_noisy = result_noisy.optimal_circuit.bind_parameters(result_noisy.optimal_parameters)
        # job = zne_estimator.run(optimal_circuit_from_noisy, operator)
        # result_zne_at_best_iter = job.result().values[0]

        ##########################################################################################
        zne_estimator = learning(BackendEstimator, processor=zne_processor, backend=FakeLima(), skip_transpile=True)
        estimator_zne_mitigated = zne_estimator(backend=FakeLima())
        history_zne_mitigated = []
        vqe = VQE(estimator=estimator_zne_mitigated, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
                  callback=lambda a, params, values, d: callback_func(history_zne_mitigated, values, params))
        result_zne_mitigated = vqe.compute_minimum_eigenvalue(operator)

        ##########################################################################################
        sep_ob_zne = True
        zne_w_readout_estimator = learning(BackendEstimator, processor=zne_w_readout_processor, backend=backend_noisy_wo_readout, skip_transpile=True)
        estimator_zne_w_readout_mitigated = zne_w_readout_estimator(backend=backend_noisy_wo_readout)
        history_zne_w_readout_mitigated = []
        vqe = VQE(estimator=estimator_zne_w_readout_mitigated, ansatz=ansatz, optimizer=optimizer, initial_point=init_pt,
                  callback=lambda a, params, values, d: callback_func(history_zne_w_readout_mitigated, values, params))
        result_zne_w_readout_mitigated = vqe.compute_minimum_eigenvalue(operator, separate_observables=sep_ob_zne)

    ##########################################################################################
        print('#' * 50)
        print("Noisy", result_noisy.optimal_value)
        print('ZNE', result_zne_mitigated.optimal_value)
        print('ZNE + Readout', result_zne_w_readout_mitigated.optimal_value)
        print("Mitigated", result_mitigated.optimal_value)
        print("Mitigated + Readout", result_mitigated_w_readout.optimal_value)
        print("Ideal", result_ideal.optimal_value)
        print("Diagonalization", min(np.real_if_close(np.linalg.eig(Operator(operator))[0])))
        print('#' * 50)

        bond_lengths.append(bond_length)
        noisy.append(result_noisy.optimal_value)
        zne.append(result_zne_mitigated.optimal_value)
        zne_w_readout.append(result_zne_w_readout_mitigated.optimal_value)
        mitigated.append(result_mitigated.optimal_value)
        mitigated_w_readout.append(result_mitigated_w_readout.optimal_value)
        ideal.append(result_ideal.optimal_value)
        diagonalization.append(min(np.real_if_close(np.linalg.eig(Operator(operator))[0])))


    plt.plot(bond_lengths, diagonalization, label='ideal')
    plt.plot(bond_lengths, mitigated, label='mitigated')
    plt.plot(bond_lengths, mitigated_w_readout, label='mitigated+readout_mitigated')
    plt.plot(bond_lengths, zne, label='zne_mitigated')
    plt.plot(bond_lengths, zne_w_readout, label='zne+readout_mitigated')
    plt.plot(bond_lengths, noisy, label='noisy')
    plt.legend()
    plt.show()

    to_save = {
        'bond_lengths': bond_lengths,
        'ideal': diagonalization,
        'mitigated': mitigated,
        'mitigated+readout': mitigated_w_readout,
        'noisy': noisy,
        'zne': zne,
        'zne+readout': zne_w_readout,
    }

    if not sep_ob_zne:
        with open('../paper_figures/vqe_with_zne_20240717.pk', 'wb') as file:
            pickle.dump(to_save, file)
    else:
        with open('../paper_figures/vqe_with_zne_20240717_sep_ob_zne.pk', 'wb') as file:
            pickle.dump(to_save, file)