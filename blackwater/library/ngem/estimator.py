"""NGEM estimator."""
from functools import wraps
from typing import Callable, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.providers import BackendV1
from qiskit.quantum_info import SparsePauliOp

from blackwater.data.generators.exp_val import ExpValueEntry
from blackwater.data.utils import (
    circuit_to_graph_data_json,
    get_backend_properties_v1,
    encode_pauli_sum_op,
)
from blackwater.exception import BlackwaterException


def patch_call(call: Callable, model: torch.nn.Module, backend: BackendV1) -> Callable:
    """

    Args:
        call: executable function
        model: pytorch mitigation model
        backend: backend

    Returns:
        patched callable funciton
    """

    @wraps(call)
    def ngem_call(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[SparsePauliOp],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:

        if len(parameter_values) > 0 and any(bool(p) for p in parameter_values):
            raise BlackwaterException("Parameters are not supported by NGEM yet.")

        result: EstimatorResult = call(
            self,
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            **run_options,
        )

        mitigated_values = []
        for value, circuit_idx, obs_idx in zip(
            result.values.tolist(), circuits, observables
        ):
            obs = self._observables[obs_idx]
            circuit = self._circuits[circuit_idx]

            if not isinstance(obs, (PauliSumOp, SparsePauliOp)):
                raise BlackwaterException(
                    "Only `PauliSumOp` observables are supported by NGEM."
                )

            graph_data = circuit_to_graph_data_json(
                circuit=circuit,
                properties=get_backend_properties_v1(backend),
                use_qubit_features=True,
                use_gate_features=True,
            )

            data = ExpValueEntry(
                circuit_graph=graph_data,
                observable=encode_pauli_sum_op(obs),
                ideal_exp_value=0.0,
                noisy_exp_value=value,
            ).to_pyg_data()

            mitigated_value = model(
                data.noisy,
                data.observable,
                data.circuit_depth,
                data.x,
                data.edge_index,
                data.batch,
            ).item()

            mitigated_values.append(mitigated_value)

        return EstimatorResult(np.array(mitigated_values), result.metadata)

    return ngem_call


def ngem(
    estimator: BaseEstimator, model: torch.nn.Module, backend: BackendV1
) -> BaseEstimator:
    """Decorator to turn Estimator into NGEM estimator.

    Args:
        estimator: estimator
        model: NGEM model
        backend: backend

    Returns:
        NGEM estimator class
    """
    estimator._call = patch_call(estimator._call, model, backend)
    return estimator
