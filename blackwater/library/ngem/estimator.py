"""NGEM estimator."""
from functools import wraps
from typing import Callable, Sequence, Tuple, List, Union, Type, Optional

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult, Estimator
from qiskit.providers import BackendV1, JobV1 as Job, Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from blackwater.data.generators.exp_val import ExpValueEntry
from blackwater.data.utils import (
    circuit_to_graph_data_json,
    get_backend_properties_v1,
    encode_pauli_sum_op,
)
from blackwater.exception import BlackwaterException


class NgemJob(Job):
    """Ngem wrapper for job results."""

    def __init__(  # pylint: disable=super-init-not-called
        self,
        base_job: Job,
        model: torch.nn.Module,
        backend: BackendV1,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        observables: Union[PauliSumOp, List[PauliSumOp]],
        parameter_values: Tuple[Tuple[float, ...], ...],
        options: Optional[Options] = None
    ) -> None:
        self._base_job: Job = base_job
        self._model = model
        self._backend = backend
        self._circuits = circuits
        self._observables = observables
        self._parameter_values = parameter_values
        self._options = options or Options()

    def result(self) -> EstimatorResult:
        result: EstimatorResult = self._base_job.result()
        properties = get_backend_properties_v1(self._backend)

        mitigated_values = []
        for value, circuit, obs, params in zip(
            result.values, self._circuits, self._observables, self._parameter_values
        ):
            if not isinstance(obs, (PauliSumOp, SparsePauliOp)):
                raise BlackwaterException(
                    "Only `PauliSumOp` observables are supported by NGEM."
                )

            bound_circuit = transpile(
                circuit, self._backend, **self._options.__dict__
            ).bind_parameters(params)

            graph_data = circuit_to_graph_data_json(
                circuit=bound_circuit,
                properties=properties,
                use_qubit_features=True,
                use_gate_features=True,
            )

            data = ExpValueEntry(
                circuit_graph=graph_data,
                observable=encode_pauli_sum_op(obs),
                ideal_exp_value=0.0,
                noisy_exp_values=[value],
            ).to_pyg_data()

            mitigated_value = self._model(
                data.noisy_0,
                data.observable,
                data.circuit_depth,
                data.x,
                data.edge_index,
                data.batch,
            ).item()

            mitigated_values.append(mitigated_value)

        return EstimatorResult(np.array(mitigated_values), result.metadata)

    def submit(self):
        return self._base_job.submit()

    def status(self):
        return self._base_job.status()

    def cancel(self):
        return self._base_job.cancel()

    def __repr__(self):
        return f"<NgemJob: {self._base_job.job_id()}>"


def patch_run(
        run: Callable,
        model: torch.nn.Module,
        backend: BackendV1,
        options: Optional[Options] = None
) -> Callable:
    """Wraps run with NGEM mitigation."""

    @wraps(run)
    def ngem_run(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        observables: Union[PauliSumOp, List[PauliSumOp]],
        parameter_values: Tuple[Tuple[float, ...], ...],
        **run_options,
    ) -> Job:
        job: Job = run(
            self,
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            **run_options,
        )
        return NgemJob(
            job,
            model=model,
            backend=backend,
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            options=options
        )

    return ngem_run


def ngem(
        cls: Type[BaseEstimator],
        model: torch.nn.Module,
        backend: BackendV1,
        options: Optional[Options] = None,
):
    """Decorator to turn Estimator into NGEM estimator.

    Args:
        cls: estimator
        model: model
        backend: backend
        options: options

    Returns:
        NGEM estimator class
    """
    new_class: type = type(f"NGEM{cls.__name__}", (cls,), {})
    new_class._run = patch_run(
        new_class._run, model, backend, options
    )  # pylint: disable=protected-access
    return new_class
