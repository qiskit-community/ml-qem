import json
import glob

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


def force_nonzero_expectation(clifford, print_bool=False):
    """Force the input Clifford operator to have a non-zero expectation value when measured in the all-Z basis.

    Args:
        clifford (Clifford): Clifford operator.
        print (bool, optional): Print the chosen random stabilizer.
    """
    # Create a QuantumCircuit from the Clifford operator
    qc_forced = clifford.to_circuit()

    # Get the stabilizers as a list of strings.
    # An example of a stabilizer string is "+XYZ"
    # with sign "+" and "Z" on qubit 1.
    stabilizers = clifford.to_dict()['stabilizer']
    for idx, stab in enumerate(stabilizers):
        # This method of forcing the Clifford operator to have
        # non-zero expectation in the all-Z basis only works
        # if the chosen stabilizer has no identity matrices.
        if 'I' not in stab:
            stabilizer = stab
            break
        # If we have tried every stabilizer, throw an exception
        if idx >= len(stabilizers) - 1:
            raise UserWarning("All of the stabilizers have the identity matrix I!")
    if print_bool:
        print(f'Stabilizer: {stabilizer}')

    # Since the Clifford circuit has no classical register, add one
    # cr = ClassicalRegister(qc_forced.num_qubits)
    # qc_forced.add_register(cr)

    # Change the measurement basis of each qubit
    for qubit in range(0, qc_forced.num_qubits):
        op = stabilizer[qc_forced.num_qubits - qubit]
        if op == 'X':
            qc_forced.append(HGate(), [[qubit]])  # Convert to x-basis measurement
        elif op == 'Y':
            qc_forced.append(SdgGate(), [[qubit]])
            qc_forced.append(HGate(), [[qubit]])
        # # Measure qubit and store in classical bit
        # if measure and op != 'I':
        #     qc_forced.measure([qubit], [qubit])

    # Compute the expectation value based on the sign of the stabilizer
    if stabilizer[0] == '+':
        expectation = 1
    elif stabilizer[0] == '-':
        expectation = -1

    return qc_forced, expectation


def construct_random_clifford(num_qubit):
    rc = random_clifford(num_qubit)

    try:
        rc_forced, _ = force_nonzero_expectation(rc)
    except UserWarning:
        rc_forced = rc.to_circuit()

    rc_forced.measure_all()
    return rc_forced, rc.to_circuit()


def cal_z_exp(counts):
    """
        Compute all sigma_z expectations values.

        Parameters
        ----------
        counts : dict
            Dictionary of state labels (keys, e.g. '000', '001')

        Returns
        -------
        z_exp : list of float
            sigma_z expectation values, where len(z_exp) is the number of qubits
        """
    shots = sum(list(counts.values()))
    num_qubits = len(list(counts.keys())[0])
    count_pos_z = np.zeros(num_qubits)  # counts of positive z
    # Convert all keys into arrays
    for key, val in counts.items():
        count_pos_z += val * np.array(list(key), dtype=int)
    count_neg_z = np.ones(num_qubits) * shots - count_pos_z  # counts of negative z
    z_exp = (count_pos_z - count_neg_z) / shots
    return z_exp


def calc_imbalance(single_z_dataset, even_qubits, odd_qubits):
    """Calculate the charge imbalance from the single-Z expectation values.

    Args:
        single_z_dataset (list[list[float]]): Single-Z expectation values. First index is qubits, second index is expectation value at each time
        even_qubits (list[int]): Indices of even qubits
        odd_qubits (list[int]): Indices of odd qubits

    Returns:
        imbalance list[float]: Charge imbalance
    """
    num_qubit = len(even_qubits) + len(odd_qubits)
    num_steps = len(single_z_dataset)
    imbalance = np.zeros(num_steps)
    for step in range(num_steps):
        ib = 0
        for qubit in range(num_qubit):
            if qubit in even_qubits:
                ib += single_z_dataset[step][qubit]
            elif qubit in odd_qubits:
                ib -= single_z_dataset[step][qubit]
            else:
                print(f'Warning: The index {i} was not in even_qubits or odd_qubits')
        imbalance[step] = ib / num_qubit
    return imbalance


def cal_all_z_exp(counts):
    """
    Compute the Z^N expectation value, where N is the number of bits in each bitstring

    Parameters
    ----------
    counts : dict
        Dictionary of state labels (keys, e.g. '000', '001') and
        counts (ints, e.g. 900, 100, 24 that add up to the total shots 1024)

    Returns
    -------
    all_z_exp : float
    """
    shots = sum(list(counts.values()))
    all_z_exp = 0
    for key, value in counts.items():
        num_ones = key.count('1')
        sign = (-1) ** (num_ones)  # Sign of the term in 'key' depends on the number of 0's, e.g. '11' is +, '110' is -
        all_z_exp += sign * value
    all_z_exp = all_z_exp / shots
    return all_z_exp


def construct_mbl_circuit(num_qubit, disorder, theta, steps):
    """Construct the circuit for Floquet dynamics of an MBL circuit.

    Args:
        num_spins (int): Number of spins. Must be even.
        W (float): Disorder strength up to np.pi.
        theta (float): Interaction strength up to np.pi.
        steps (int): Number of steps.
    """
    qc = QuantumCircuit(num_qubit)

    # Hard domain wall initial state
    # Qubits 0 to num_qubit/2 - 1 are up, and qubits num_qubit/2 to num_qubit - 1 are down
    for qubit_idx in range(num_qubit):
        if qubit_idx % 2 == 1:
            qc.x(qubit_idx)

    ## Floquet evolution
    for step in range(steps):
        # Interactions between even layers
        for even_qubit in range(0, num_qubit, 2):
            qc.append(CZGate(), (even_qubit, even_qubit + 1))
            qc.append(U3Gate(theta, 0, -np.pi), [even_qubit])
            qc.append(U3Gate(theta, 0, -np.pi), [even_qubit + 1])
        # Interactions between odd layers
        for odd_qubit in range(1, num_qubit - 1, 2):
            qc.append(CZGate(), (odd_qubit, odd_qubit + 1))
            qc.append(U3Gate(theta, 0, -np.pi), [odd_qubit])
            qc.append(U3Gate(theta, 0, -np.pi), [odd_qubit + 1])
        # Apply RZ disorder
        for q in range(num_qubit):
            qc.append(PhaseGate(disorder[q]), [q])

    # Measure Z^{\otimes num_qubit}, or the all-Z operator from which all Z, ZZ, ... operators can be computed
    qc.measure_all()

    return qc


def generate_disorder(n_qubits, disorder_strength=np.pi, seed=0):
    """Generate disorder

    Args:
        n_qubits (int): Number of qubits
        disorder_strength (float, optional): Scales disorder strength from min/max of -pi/pi. Defaults to pi.

    Returns:
        List[float]: List of angles in single-qubit phase gates that correspond to disorders
    """
    random.seed(seed)
    disorder = [np.random.uniform(-1 * disorder_strength, disorder_strength) for _ in range(n_qubits)]
    return disorder