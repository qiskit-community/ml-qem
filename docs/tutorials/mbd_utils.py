import json
import glob
import copy

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

from qiskit.quantum_info import random_clifford, Clifford

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

import numpy as np

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import (
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    CXGate,
    CYGate,
    CZGate,
    SwapGate,
)
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RXGate, ZGate, IGate, YGate, CXGate
from qiskit_aer.noise import depolarizing_error, coherent_unitary_error, NoiseModel
from qiskit.result import marginal_counts


def modify_and_add_noise_to_model(
        theta: float = np.pi / 8,
        FakeBackend=FakeLima(),
        Simulator=AerSimulator
) -> NoiseModel:
    """
    Modify a FakeBackend with by replacing its CX noise model with the 'add_coherent_cx_noise_to_model' function
    """
    noise_model = NoiseModel.from_backend(FakeBackend)
    noise_dict = noise_model.to_dict()['errors']

    ind_to_del = []
    for i in range(len(noise_dict)):
        if noise_dict[i]['operations'] == ['cx']:
            ind_to_del += [i]

    for i in sorted(ind_to_del, reverse=True):
        del noise_dict[i]

    new_noise_model = NoiseModel.from_dict({'errors': noise_dict})
    new_noise_model = add_coherent_cx_noise_to_model(new_noise_model, theta=theta)
    modified_backend = Simulator.from_backend(FakeBackend, noise_model=new_noise_model)

    return modified_backend, new_noise_model


def add_coherent_cx_noise_to_model(
        noise_model: NoiseModel,
        theta: float = 0,
) -> NoiseModel:
    """
    Add a noise channel with over-rotation on the CNOT gate on the target by theta to the backend
    """
    cnot = Operator(CXGate())
    up_state = 0.5 * (Operator(IGate()) + Operator(ZGate()))
    down_state = 0.5 * (Operator(IGate()) - Operator(ZGate()))
    over_rotated_cnot = Operator(IGate()).tensor(up_state) + 1.0j * Operator(RXGate(np.pi + theta)).tensor(down_state)
    err_unitary = over_rotated_cnot @ cnot
    err_coherent = coherent_unitary_error(err_unitary)

    noise_model.add_all_qubit_quantum_error(err_coherent, ['cx'])

    return noise_model


def random_clifford_circuit(num_qubits, depth, max_operands=2, reset=False, seed=None):
    """Generate random circuit of arbitrary size and form.

    This function will generate a random circuit by randomly selecting gates
    from the set of Clifford gates.

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum operands of each gate (between 1 and 3)
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
    """
    if max_operands < 1 or max_operands > 2:
        raise CircuitError("max_operands must be 1 or 2")

    one_q_ops = [
        IGate,
        XGate,
        YGate,
        ZGate,
        HGate,
        SGate,
        SdgGate,
    ]
    one_param = []
    two_param = []
    three_param = []
    two_q_ops = [CXGate, CYGate, CZGate, SwapGate]
    three_q_ops = []

    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            operands = [remaining_qubits.pop() for _ in range(num_operands)]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            register_operands = [qr[i] for i in operands]
            op = operation()

            qc.append(op, register_operands)

    return qc


def force_nonzero_expectation_from_clifford_circuit(clifford_circuit, print_bool=False):
    """Force the input Clifford `QuantumCircuit` to have a non-zero expectation value when measured in the all-Z basis.

    Args:
        clifford (QuantumCircuit): Clifford as a QuantumCircuit.
        print (bool, optional): Print the chosen random stabilizer.
    """
    # Convert the Clifford circuit into a `Clifford` object
    clifford = Clifford(clifford_circuit)
    # Copy the Clifford circuit into the quantum circuit that will be returned
    qc_forced = copy.deepcopy(clifford_circuit)

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


def construct_random_clifford(num_qubit, depth, max_operands=2):
    rc = random_clifford_circuit(num_qubit, depth, max_operands=max_operands)
    enforced = True

    try:
        rc_forced, _ = force_nonzero_expectation_from_clifford_circuit(rc)
    except UserWarning:
        rc_forced = rc
        enforced = False

    rc_forced.measure_all()
    return rc_forced, enforced


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

    # for step in range(num_steps):
    #     ib = 0
    #     for qubit in range(num_qubit):
    #         if qubit in even_qubits:
    #             ib += single_z_dataset[step][qubit]
    #         elif qubit in odd_qubits:
    #             ib -= single_z_dataset[step][qubit]
    #         else:
    #             print(f'Warning: The index {ib} was not in even_qubits or odd_qubits')
    #     imbalance[step] = ib / num_qubit
    # return imbalance

    density = (1 - np.array(single_z_dataset)) / 2
    N_odd = np.sum(density[:, np.array(odd_qubits, dtype=int)], 1)
    N_even = np.sum(density[:, np.array(even_qubits, dtype=int)], 1)
    return (N_odd - N_even) / (N_even + N_odd)


def cal_all_z_exp(counts, marginal_over=None):
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
    if marginal_over:
        counts = marginal_counts(counts, indices=marginal_over)

    shots = sum(list(counts.values()))
    all_z_exp = 0
    for key, value in counts.items():
        num_ones = key.count('1')
        sign = (-1) ** num_ones  # Sign of the term in 'key' depends on the number of 0's, e.g. '11' is +, '110' is -
        all_z_exp += sign * value
    all_z_exp = all_z_exp / shots

    return all_z_exp


def construct_mbl_circuit(num_qubit, disorder, theta, steps, completely_random=False):
    """Construct the circuit for Floquet dynamics of an MBL circuit.

    Args:
        num_spins (int): Number of spins. Must be even.
        W (float): Disorder strength up to np.pi.
        theta (float): Interaction strength up to np.pi.
        steps (int): Number of steps.
        broken_connections (list[tuple]): Qubit pairs that are should not interact in the MBL circuit written as (i, j) where i < j
    """
    qc = QuantumCircuit(num_qubit)

    # Hard domain wall initial state
    # Qubits 0 to num_qubit/2 - 1 are up, and qubits num_qubit/2 to num_qubit - 1 are down
    for qubit_idx in range(num_qubit):
        if qubit_idx % 2 == 1:
            qc.x(qubit_idx)

    ## Floquet evolution
    if not completely_random:
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
    else:
        for step in range(steps):
            # Interactions between even layers
            for even_qubit in range(0, num_qubit, 2):
                qc.append(CZGate(), (even_qubit, even_qubit + 1))
                qc.append(U3Gate(*gen_random_param(3)), [even_qubit])
                qc.append(U3Gate(*gen_random_param(3)), [even_qubit + 1])
            # Interactions between odd layers
            for odd_qubit in range(1, num_qubit - 1, 2):
                qc.append(CZGate(), (odd_qubit, odd_qubit + 1))
                qc.append(U3Gate(*gen_random_param(3)), [odd_qubit])
                qc.append(U3Gate(*gen_random_param(3)), [odd_qubit + 1])
            for q in range(num_qubit):
                qc.append(PhaseGate(*gen_random_param(1)), [q])

    # Measure Z^{\otimes num_qubit}, or the all-Z operator from which all Z, ZZ, ... operators can be computed
    qc.measure_all()

    return qc


def gen_random_param(size):
    return 8 * np.pi * np.random.random(size) - 4 * np.pi


def generate_disorder(n_qubits, disorder_strength=np.pi, seed=None):
    """Generate disorder

    Args:
        n_qubits (int): Number of qubits
        disorder_strength (float, optional): Scales disorder strength from min/max of -pi/pi. Defaults to pi.

    Returns:
        List[float]: List of angles in single-qubit phase gates that correspond to disorders
    """
    if seed is not None: np.random.seed(seed)
    disorder = [np.random.uniform(-1 * disorder_strength, disorder_strength) for _ in range(n_qubits)]
    return disorder


def construct_mbl_circ_with_cut(num_qubit, disorder, theta, steps, broken_connections=None):
    """Construct the circuit for Floquet dynamics of an MBL circuit.

    Args:
        num_spins (int): Number of spins. Must be even.
        W (float): Disorder strength up to np.pi.
        theta (float): Interaction strength up to np.pi.
        steps (int): Number of steps.
        broken_connections (list[tuple]): Qubit pairs that are should not interact in the MBL circuit written as (i, j) where i < j
    """
    if broken_connections is None:
        broken_connections = []

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
            if (even_qubit, even_qubit + 1) not in broken_connections:
                qc.append(CZGate(), (even_qubit, even_qubit + 1))
            qc.append(U3Gate(theta, 0, -np.pi), [even_qubit])
            qc.append(U3Gate(theta, 0, -np.pi), [even_qubit + 1])
        # Interactions between odd layers
        for odd_qubit in range(1, num_qubit - 1, 2):
            if (odd_qubit, odd_qubit + 1) not in broken_connections:
                qc.append(CZGate(), (odd_qubit, odd_qubit + 1))
            qc.append(U3Gate(theta, 0, -np.pi), [odd_qubit])
            qc.append(U3Gate(theta, 0, -np.pi), [odd_qubit + 1])
        # Apply RZ disorder
        for q in range(num_qubit):
            qc.append(PhaseGate(disorder[q]), [q])

    # Measure Z^{\otimes num_qubit}, or the all-Z operator from which all Z, ZZ, ... operators can be computed
    qc.measure_all()

    return qc


if __name__ == '__main__':
    qc = construct_mbl_circ_with_cut(8, generate_disorder(8), 0.1*np.pi, 2, [(1, 2), (5, 6)])
    qc.draw('mpl', fold=-1).show()

    # num_steps = 0
    # num_qubit = 8
    # rb = construct_mbl_circuit(num_qubit, [], 0, num_steps, completely_random=True)
    # print(rb.draw(fold=-1, idle_wires=False))
