import qiskit, random, os
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer.noise import depolarizing_error, coherent_unitary_error, mixed_unitary_error, thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel, mixed_unitary_error, pauli_error
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import CXGate, RXGate, IGate, ZGate, RZXGate, RZZGate
from qiskit.providers.fake_provider import FakeMontreal, FakeLima
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)


def fix_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    print(f'random seed fixed to {seed}')


class RemoveReadoutErrors:
    def __init__(self, backend=FakeLima(), simulator=AerSimulator):
        self.backend = backend
        self.simulator = simulator
        self.num_qubits = self.backend.configuration().n_qubits

    def remove_readout_errors(self) -> NoiseModel:
        noise_model = NoiseModel.from_backend(self.backend)
        noise_dict = noise_model.to_dict()['errors']

        ind_to_del = []
        for i in range(len(noise_dict)):
            if noise_dict[i]['operations'] == ['measure']:
                ind_to_del += [i]

        for i in sorted(ind_to_del, reverse=True):
            del noise_dict[i]

        new_noise_model = NoiseModel.from_dict({'errors': noise_dict})
        modified_backend = self.simulator.from_backend(self.backend, noise_model=new_noise_model)

        return modified_backend, new_noise_model


class AddNoise:
    def __init__(self, backend=FakeLima(), simulator=AerSimulator):
        self.uniform = None
        self.theta = None
        self.add_depolarization = None
        self.add_coherent = None
        self.backend = backend
        self.simulator = simulator
        self.num_qubits = self.backend.configuration().n_qubits
        self.cnot_directions = [list(pair) for pair in self.backend.configuration().coupling_map]

        self.cnot = Operator(CXGate())
        self.up_state = 0.5 * (Operator(IGate()) + Operator(ZGate()))
        self.down_state = 0.5 * (Operator(IGate()) - Operator(ZGate()))

    def add_coherent_noise(self, theta, uniform: bool = False, add_depolarization: bool = True,
                           seed: int = None, add_coherent: bool = True) -> NoiseModel:
        self.theta = theta
        self.uniform = uniform
        self.add_depolarization = add_depolarization
        self.add_coherent = add_coherent
        if seed is not None: fix_random_seed(seed)

        noise_model = NoiseModel.from_backend(self.backend)
        noise_dict = noise_model.to_dict()['errors']

        ind_to_del = []
        for i in range(len(noise_dict)):
            if noise_dict[i]['operations'] == ['cx']:
                ind_to_del += [i]

        for i in sorted(ind_to_del, reverse=True):
            del noise_dict[i]

        new_noise_model = NoiseModel.from_dict({'errors': noise_dict})
        if self.add_coherent:
            new_noise_model = self.cx_over_rotation(new_noise_model)
        else:
            new_noise_model = self.restore_incoherent(new_noise_model)
        modified_backend = self.simulator.from_backend(self.backend, noise_model=new_noise_model)

        return modified_backend, new_noise_model

    def get_over_rotated_cx_uni_err(self, theta):
        over_rotated_cnot = Operator(IGate()).tensor(self.up_state) + 1.0j * Operator(RXGate(np.pi + theta)).tensor(
            self.down_state)
        err_unitary = over_rotated_cnot @ self.cnot
        return coherent_unitary_error(err_unitary)

    def get_depol_therm_error(self, qubits):
        depol_mag = self.backend.properties().gate_error('cx', qubits)
        gate_time = self.backend.properties().gate_length('cx', qubits)
        depol_err = depolarizing_error(depol_mag, 2)
        t1 = self.backend.properties().t1(qubits[0])
        t2 = self.backend.properties().t2(qubits[0])
        therm_err_q0 = thermal_relaxation_error(t1=t1, t2=t2, time=gate_time)
        t1 = self.backend.properties().t1(qubits[1])
        t2 = self.backend.properties().t2(qubits[1])
        therm_err_q1 = thermal_relaxation_error(t1=t1, t2=t2, time=gate_time)
        return depol_err, therm_err_q0, therm_err_q1

    def cx_over_rotation(self, noise_model: NoiseModel) -> NoiseModel:
        if self.uniform:
            coherent_err = self.get_over_rotated_cx_uni_err(self.theta)
            if self.add_depolarization:
                depol_err, therm_err_q0, therm_err_q1 = self.get_depol_therm_error(self.cnot_directions[0])
                composite_error = coherent_err.compose(depol_err).compose(therm_err_q0).compose(therm_err_q1)
                noise_model.add_all_qubit_quantum_error(composite_error, ['cx'])
            else:
                noise_model.add_all_qubit_quantum_error(coherent_err, ['cx'])
        else:
            thetas = np.random.uniform(0, self.theta, size=len(self.cnot_directions))
            print('thetas', thetas)
            for pair, theta in zip(self.cnot_directions, thetas):
                coherent_err = self.get_over_rotated_cx_uni_err(theta)
                if self.add_depolarization:
                    depol_err, therm_err_q0, therm_err_q1 = self.get_depol_therm_error(pair)
                    composite_error = coherent_err.compose(depol_err).compose(therm_err_q0).compose(therm_err_q1)
                    noise_model.add_quantum_error(composite_error, 'cx', pair)
                else:
                    noise_model.add_quantum_error(coherent_err, 'cx', pair)

        return noise_model

    def restore_incoherent(self, noise_model: NoiseModel) -> NoiseModel:
        for pair in self.cnot_directions:
            depol_err, therm_err_q0, therm_err_q1 = self.get_depol_therm_error(pair)
            composite_error = depol_err.compose(therm_err_q0).compose(therm_err_q1)
            noise_model.add_quantum_error(composite_error, 'cx', pair)

        return noise_model
