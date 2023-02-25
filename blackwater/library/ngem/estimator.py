"""NGEM estimator."""
from functools import wraps
from typing import Callable, Tuple, List, Union, Type, Optional

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.providers import JobV1 as Job, Options, BackendV2
from qiskit.quantum_info import SparsePauliOp

from blackwater.data import ExpValData
from blackwater.exception import BlackwaterException


class NgemJob(Job):
    """Ngem wrapper for job results."""

    def __init__(  # pylint: disable=super-init-not-called
        self,
        base_job: Job,
        model: torch.nn.Module,
        backend: BackendV2,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        observables: Union[PauliSumOp, List[PauliSumOp]],
        parameter_values: Tuple[Tuple[float, ...], ...],
        skip_transpile: bool,
        options: Optional[Options] = None,
    ) -> None:
        self._base_job: Job = base_job
        self._model = model
        self._backend = backend
        self._circuits = circuits
        self._observables = observables
        self._parameter_values = parameter_values
        self._options = options or Options()
        self._skip_transpile = skip_transpile

    def result(self) -> EstimatorResult:
        result: EstimatorResult = self._base_job.result()

        mitigated_values = []
        for value, circuit, obs, params in zip(
            result.values, self._circuits, self._observables, self._parameter_values
        ):
            if not isinstance(obs, (PauliSumOp, SparsePauliOp)):
                raise BlackwaterException(
                    "Only `PauliSumOp` observables are supported by NGEM."
                )

            if self._skip_transpile:
                bound_circuit = circuit.bind_parameters(params)
            else:
                bound_circuit = transpile(
                    circuit, self._backend, **self._options.__dict__
                ).bind_parameters(params)

            data = ExpValData.build(
                circuit=bound_circuit,
                expectation_values=[value],
                observable=obs,
                backend=self._backend,
            ).to_pyg()

            noisy_exp_val, _ = torch.tensor_split(  # pylint: disable=no-member
                data.y, 2, dim=1
            )
            mitigated_value = self._model(
                data.x,
                data.edge_index,
                noisy_exp_val,
                data.observable,
                data.circuit_depth,
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
    backend: BackendV2,
    skip_transpile: bool,
    options: Optional[Options] = None,
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
            skip_transpile=skip_transpile,
            options=options,
        )

    return ngem_run


# pylint: disable=protected-access
def ngem(
    cls: Type[BaseEstimator],
    model: torch.nn.Module,
    backend: BackendV2,
    skip_transpile: bool = False,
    options: Optional[Options] = None,
):
    """Decorator to turn Estimator into NGEM estimator.

    Args:
        cls: estimator
        model: model
        backend: backend
        skip_transpile: skip transpilation
        options: options

    Returns:
        NGEM estimator class
    """
    new_class: type = type(f"NGEM{cls.__name__}", (cls,), {})
    new_class._run = patch_run(  # type: ignore[attr-defined]
        new_class._run,  # type: ignore[attr-defined]
        model,
        backend,
        skip_transpile,
        options,
    )
    return new_class
