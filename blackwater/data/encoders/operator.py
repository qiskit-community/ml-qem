"""Operator encoders."""
from dataclasses import dataclass
from typing import Union, List

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from blackwater.exception import BlackwaterException


@dataclass
class OperatorData:
    operator: List[List[float]]


def encode_pauli_sum_operator(operator: PauliSumOp) -> List[List[float]]:
    mapping = {
        "X": [0, 0, 0, 1],
        "Y": [0, 0, 1, 0],
        "Z": [0, 1, 0, 0],
        "I": [1, 0, 0, 0],
    }
    coeffs = [k.coeffs[0].real for k in operator]
    strings = [str(k.primitive.paulis[0]) for k in operator]
    rows = []
    for c, pauli in zip(coeffs, strings):
        encoded_row = [c]
        for p in pauli:
            encoded_row += mapping.get(p, [0, 0, 0, 0])
        rows.append(encoded_row)
    return rows


def encode_sparse_pauli_operatpr(operator: SparsePauliOp) -> List[List[float]]:
    return encode_pauli_sum_operator(PauliSumOp.from_list(operator.to_list()))


def encode_operator(operator: Union[BaseOperator]) -> OperatorData:
    if isinstance(operator, SparsePauliOp):
        result = encode_sparse_pauli_operatpr(operator)
    elif isinstance(operator, PauliSumOp):
        result = encode_pauli_sum_operator(operator)
    else:
        raise BlackwaterException(
            f"Operator of type [{type(operator)}] is not supported yet."
        )

    return OperatorData(result)
