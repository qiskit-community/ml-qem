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

backend = FakeLima()
backend_noisy = AerSimulator.from_backend(backend)  # Noisy
run_config_noisy = {'shots': 10000, 'backend': backend_noisy, 'name': 'noisy'}
qasm_sim = QasmSimulator()

NUM_QUBITS = 4
N = 10


def get_all_z_exp_wo_shot_noise(circuit, marginal_over=None):
    circuit_copy = circuit.copy()
    circuit_copy.remove_final_measurements()
    circuit_copy.save_density_matrix()

    def int_to_bin(n, num_bits):
        if n < 2 ** num_bits:
            binary_str = bin(n)[2:]
            return binary_str.zfill(num_bits)
        else:
            raise ValueError

    circuit_copy = transpile(circuit_copy, backend=backend_noisy, optimization_level=3)
    job = qasm_sim.run(circuit_copy)
    # job = execute(circuit_copy, QasmSimulator(), backend_options={'method': 'statevector'})
    probs = np.real(np.diag(job.result().results[0].data.density_matrix))
    probs = {int_to_bin(i, NUM_QUBITS): p for i, p in enumerate(probs)}

    if marginal_over:
        probs = marginal_counts(probs, indices=marginal_over)

    exp_val = 0
    for key, prob in probs.items():
        num_ones = key.count('1')
        exp_val += (-1) ** num_ones * prob

    return exp_val


def get_noisy_exp_vals(args):
    i, trans_circuit, obs = args
    obs = np.array(list(obs))
    marginal_over = np.where(obs == 'I')[0].tolist() if 'I' in obs else None
    job_noisy = execute(trans_circuit, **run_config_noisy)
    counts_noisy = job_noisy.result().get_counts()
    return i, cal_all_z_exp(counts_noisy, marginal_over=marginal_over)


def get_ideal_exp_vals(args):
    i, circuit, obs = args
    # print(obs)
    obs = np.array(list(obs))
    marginal_over = np.where(obs == 'I')[0].tolist() if 'I' in obs else None
    # print(marginal_over)
    return i, get_all_z_exp_wo_shot_noise(circuit, marginal_over=marginal_over)


def transpile_circuits(args):
    i, circuit = args
    return i, transpile(circuit, backend=backend_noisy, optimization_level=3)


if __name__ == '__main__':
    np.random.seed(0)
    pauli_list_full = [''.join(s) for s in itertools.product(['X', 'Y', 'Z', 'I'], repeat=NUM_QUBITS)]
    np.random.shuffle(pauli_list_full)
    print(pauli_list_full)
    pauli_list_full_tiled = np.repeat(pauli_list_full, N).tolist()
    print(len(pauli_list_full_tiled))

    circuits = []
    for pauli in tqdm(pauli_list_full):
        ansatz = TwoLocal(num_qubits=NUM_QUBITS, rotation_blocks='rx', entanglement_blocks='cx', reps=3)
        measurement_circ = QuantumCircuit(NUM_QUBITS)
        for i, p in enumerate(pauli):
            if p in ['Z', 'I']:
                pass
            elif p == 'X':
                measurement_circ.h(i)
            elif p == 'Y':
                measurement_circ.sdg(i)
                measurement_circ.h(i)
            else:
                raise Exception
        ansatz.compose(measurement_circ, inplace=True)
        ansatz.measure_all()
        for _ in range(N):
            circuits.append(ansatz.bind_parameters(np.random.uniform(-5, 5, ansatz.num_parameters)))

    print(len(circuits))

    ###############################################################################
    # Can't do it before get_ideal_exp_vals because it appends the save_density_matrix to ALL qubits on the transpiled circuits
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
    noisy_exp_vals = np.zeros((len(trans_circuits), 1))
    assert len(trans_circuits) == len(pauli_list_full_tiled)
    iterable = [(i, trans_circ, obs) for i, (trans_circ, obs) in enumerate(zip(trans_circuits, pauli_list_full_tiled))]
    iterable = tqdm(iterable, total=len(circuits), desc="Processing")
    with Pool() as pool:
        results = pool.map(get_noisy_exp_vals, iterable)
    for i, val in results:
        noisy_exp_vals[i] = val

    # for ideal, noisy in zip(ideal_exp_vals, noisy_exp_vals):
    #     print(ideal, noisy)

    entries = []
    for trans_circ, ideal, noisy, obs in tqdm(zip(trans_circuits, ideal_exp_vals.tolist(), noisy_exp_vals.tolist(), pauli_list_full_tiled)):
        to_append = {
            'circuit': trans_circ,
            'ideal_exp_value': ideal,
            'noisy_exp_values': noisy,
            'meas_basis': obs,
        }
        entries.append(to_append)
    print(len(entries))

    with open(f'./data/vqe/two_local_{NUM_QUBITS}q_3reps.pk', 'wb') as file:
        pickle.dump(entries, file)
