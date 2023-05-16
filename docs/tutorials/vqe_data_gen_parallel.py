from tqdm import tqdm
import itertools
import pickle
from multiprocessing import Pool

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import execute
from qiskit.circuit.library import TwoLocal
from qiskit.providers.fake_provider import FakeLima
from qiskit.result import marginal_counts
from qiskit_aer import AerSimulator
from qiskit_aer import QasmSimulator
from tqdm import tqdm

from mbd_utils import cal_all_z_exp

from qiskit.primitives import BackendEstimator, Estimator

# backend_noisy = AerSimulator.from_backend(backend)  # Noisy
# run_config_noisy = {'shots': 10000, 'backend': backend_noisy, 'name': 'noisy'}
# qasm_sim = QasmSimulator()

backend = FakeLima()
estimator_noisy = BackendEstimator(backend)
estimator_noisy.set_transpile_options(optimization_level=0)
estimator_ideal = Estimator()
num_shots = 10000

NUM_QUBITS = 3
N = 5000


def get_all_z_exp_wo_shot_noise(circuit, observable):
    return estimator_ideal.run(circuit, observables=observable).result().values.item()


# def get_noisy_exp_vals(args):
#     i, trans_circuit, obs = args
#     obs = np.array(list(obs))
#     marginal_over = np.where(obs == 'I')[0].tolist() if 'I' in obs else None
#     job_noisy = execute(trans_circuit, **run_config_noisy)
#     counts_noisy = job_noisy.result().get_counts()
#     return i, cal_all_z_exp(counts_noisy, marginal_over=marginal_over)


def get_noisy_exp_vals(args):
    i, circuit, obs = args
    return i, estimator_noisy.run(circuit, num_shots=num_shots, observables=obs).result().values.item()


# def get_ideal_exp_vals(args):
#     i, circuit, obs = args
#     # print(obs)
#     obs = np.array(list(obs))
#     marginal_over = np.where(obs == 'I')[0].tolist() if 'I' in obs else None
#     # print(marginal_over)
#     return i, get_all_z_exp_wo_shot_noise(circuit, marginal_over=marginal_over)


def get_ideal_exp_vals(args):
    i, circuit, obs = args
    return i, get_all_z_exp_wo_shot_noise(circuit, observable=obs)


def transpile_circuits(args):
    i, circuit = args
    return i, transpile(circuit, backend=backend, optimization_level=0)


if __name__ == '__main__':
    np.random.seed(0)
    # pauli_list_full = [''.join(s) for s in itertools.product(['X', 'Y', 'Z', 'I'], repeat=NUM_QUBITS)]
    # pauli_list_full.remove('I' * NUM_QUBITS)
    # np.random.shuffle(pauli_list_full)
    pauli_list_full = ['XXX', 'ZZZ', 'YYY']
    print(pauli_list_full)
    pauli_list_full_tiled = np.repeat(pauli_list_full, N).tolist()
    print(len(pauli_list_full_tiled))

    circuits = []
    for pauli in tqdm(pauli_list_full):
        ansatz = TwoLocal(num_qubits=NUM_QUBITS, rotation_blocks='ry', entanglement_blocks='cz', reps=3)
        # measurement_circ = QuantumCircuit(NUM_QUBITS)
        # for i, p in enumerate(pauli):
        #     if p in ['Z', 'I']:
        #         pass
        #     elif p == 'X':
        #         measurement_circ.h(i)
        #     elif p == 'Y':
        #         measurement_circ.sdg(i)
        #         measurement_circ.h(i)
        #     else:
        #         raise Exception
        # ansatz.compose(measurement_circ, inplace=True)
        # ansatz.measure_all()
        for _ in range(N):
            circuits.append(ansatz.bind_parameters(np.random.uniform(-5, 5, ansatz.num_parameters)))

    print(len(circuits))

    ###############################################################################
    # Can't do it after get_ideal_exp_vals because it appends the save_density_matrix to ALL qubits on the transpiled circuits
    trans_circuits = [None] * len(circuits)
    iterable = [(i, circ) for i, circ in enumerate(circuits)]
    iterable = tqdm(iterable, total=len(circuits), desc="Transpiling")
    with Pool() as pool:
        results = pool.map(transpile_circuits, iterable)
    for i, val in results:
        trans_circuits[i] = val

    ###############################################################################
    ideal_exp_vals = np.zeros((len(circuits), 1))
    assert len(circuits) == len(pauli_list_full_tiled)
    iterable = [(i, circ, obs) for i, (circ, obs) in enumerate(zip(circuits, pauli_list_full_tiled))]
    iterable = tqdm(iterable, total=len(circuits), desc="Processing")
    with Pool() as pool:
        results = pool.map(get_ideal_exp_vals, iterable)
    for i, val in results:
        ideal_exp_vals[i] = val

    ###############################################################################
    noisy_exp_vals = np.zeros((len(circuits), 1))
    assert len(circuits) == len(pauli_list_full_tiled)
    iterable = [(i, circ, obs) for i, (circ, obs) in enumerate(zip(circuits, pauli_list_full_tiled))]
    iterable = tqdm(iterable, total=len(circuits), desc="Processing")
    with Pool() as pool:
        results = pool.map(get_noisy_exp_vals, iterable)
    for i, val in results:
        noisy_exp_vals[i] = val

    entries = []
    for trans_circ, circ, ideal, noisy, obs in tqdm(zip(trans_circuits, circuits, ideal_exp_vals.tolist(), noisy_exp_vals.tolist(), pauli_list_full_tiled)):
        to_append = {
            'trans_circuit': trans_circ,
            'circuit': circ,
            'ideal_exp_value': ideal,
            'noisy_exp_values': noisy,
            'meas_basis': obs,
        }
        entries.append(to_append)
    print(len(entries))

    with open(f'./data/vqe/two_local_{NUM_QUBITS}q_3reps_oplev0_rycz.pk', 'wb') as file:
        pickle.dump(entries, file)
