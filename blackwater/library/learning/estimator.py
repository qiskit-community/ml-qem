"""Learning based estimator."""
from functools import wraps
from typing import Union, Tuple, List, Any, Type, Optional, Callable

import numpy as np
import torch.nn
from qiskit import QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.providers import JobV1 as Job, Options, Backend, BackendV1
from qiskit.quantum_info import SparsePauliOp
from sklearn.base import BaseEstimator as ScikitBaseEstimator

from blackwater.data.utils import get_backend_properties_v1, encode_pauli_sum_op
from blackwater.exception import BlackwaterException
from blackwater.library.temp import encode_data


class LearningMethodEstimatorProcessor:
    def process(
            self,
            expectation_value: np.ndarray[Any, np.dtype[np.float64]],
            circuits: Union[QuantumCircuit, List[QuantumCircuit]],
            observables: Union[PauliSumOp, List[PauliSumOp]],
            parameter_values: Tuple[Tuple[float, ...], ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        raise NotImplementedError


class ScikitLearningModelProcessor(LearningMethodEstimatorProcessor):
    def __init__(self,
                 model: ScikitBaseEstimator,
                 backend: BackendV1):
        self._model = model
        self._backend = backend
        self._properties = get_backend_properties_v1(backend)

    def process(
            self,
            expectation_value: np.ndarray[Any, np.dtype[np.float64]],
            circuits: Union[QuantumCircuit, List[QuantumCircuit]],
            observables: Union[PauliSumOp, List[PauliSumOp]],
            parameter_values: Tuple[Tuple[float, ...], ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:

        results = []
        for p in observables:
            coeff = p.coeffs
            pauli = p.paulis

            model_input, _ = encode_data(
                circuits=[circuits],
                properties=self._properties,
                ideal_exp_vals=[[0.]],
                noisy_exp_vals=[[expectation_value]],
                num_qubits=1,
                meas_bases=encode_pauli_sum_op(SparsePauliOp(pauli))
            )

            output = self._model.predict(model_input).item()

            results.append(output * coeff[0])

        return np.sum(results)


class TorchLearningModelProcessor(LearningMethodEstimatorProcessor):

    def __init__(self,
                 model: torch.nn.Module,
                 backend: BackendV1
                 ):
        self._model = model
        self._backend = backend
        self._properties = get_backend_properties_v1(backend)

    def process(self,
                expectation_value: np.ndarray[Any, np.dtype[np.float64]],
                circuits: Union[QuantumCircuit, List[QuantumCircuit]],
                observables: Union[PauliSumOp, List[PauliSumOp]],
                parameter_values: Tuple[Tuple[float, ...], ...]
                ) -> np.ndarray[Any, np.dtype[np.float64]]:

        results = []

        for p in observables:
            coeff = p.coeffs
            pauli = p.paulis

            model_input, _ = encode_data(
                circuits=[circuits],
                properties=self._properties,
                ideal_exp_vals=[[0.]],
                noisy_exp_vals=[[expectation_value]],
                num_qubits=1,
                meas_bases=encode_pauli_sum_op(SparsePauliOp(pauli))
            )

            output = self._model(model_input).item()

            results.append(output * coeff[0])

        return np.sum(results)


class EmptyProcessor(LearningMethodEstimatorProcessor):
    def process(self, expectation_value: np.ndarray[Any, np.dtype[np.float64]],
                circuits: Union[QuantumCircuit, List[QuantumCircuit]], observables: Union[PauliSumOp, List[PauliSumOp]],
                parameter_values: Tuple[Tuple[float, ...], ...]) -> np.ndarray[Any, np.dtype[np.float64]]:
        return expectation_value


class PostProcessedJob(Job):
    """PostProcessedJob."""

    def __init__(self, base_job: Job, processor: LearningMethodEstimatorProcessor,
                 circuits: Union[QuantumCircuit, List[QuantumCircuit]],
                 observables: Union[PauliSumOp, List[PauliSumOp]], parameter_values: Tuple[Tuple[float, ...], ...],
                 skip_transpile: bool, backend: Optional[Backend], job_id: str, options: Optional[Options] = None,
                 **kwargs) -> None:

        super().__init__(backend, job_id, **kwargs)
        self._base_job: Job = base_job
        self._processor = processor
        self._circuits = circuits
        self._observables = observables
        self._parameter_values = parameter_values
        self._options = options or Options()
        self._skip_transpile = skip_transpile

    def result(self) -> EstimatorResult:
        result: EstimatorResult = self._base_job.result()

        mitigated_values = []
        metadata = []
        for value, circuit, obs, params, meta in zip(
                result.values, self._circuits,
                self._observables, self._parameter_values,
                result.metadata
        ):
            if not isinstance(obs, (PauliSumOp, SparsePauliOp)):
                raise BlackwaterException(
                    "Only `PauliSumOp` observables are supported by learning primitive."
                )

            if self._skip_transpile:
                bound_circuit = circuit.bind_parameters(params)
            else:
                bound_circuit = transpile(
                    circuit, self.backend(), **self._options.__dict__
                ).bind_parameters(params)

            mitigated_value = self._processor.process(
                expectation_value=value,
                circuits=bound_circuit,
                observables=obs,
                parameter_values=params
            )

            mitigated_values.append(mitigated_value)
            metadata.append({**meta, "original_value": value})

        return EstimatorResult(np.array(mitigated_values), metadata)

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
        processor: LearningMethodEstimatorProcessor,
        skip_transpile: bool,
        backend: Optional[Backend] = None,
        options: Optional[Options] = None,
) -> Callable:
    """Wraps run with NGEM mitigation."""

    @wraps(run)
    def patched_run(
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
        return PostProcessedJob(
            job,
            job_id=job.job_id(),
            backend=backend,
            processor=processor,
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            skip_transpile=skip_transpile,
            options=options,
        )

    return patched_run


def learning(
        cls: Type[BaseEstimator],
        processor: LearningMethodEstimatorProcessor,
        skip_transpile: bool = False,
        backend: Optional[Backend] = None,
        options: Optional[Options] = None,
):
    """Decorator to turn Estimator into LearningEstimator.

    Args:
        cls: estimator
        processor: processor implementation of learning based method
        backend: optional backend
        skip_transpile: skip transpilation
        options: options

    Returns:
        learning estimator class
    """
    new_class: type = type(f"Learning{cls.__name__}", (cls,), {})
    new_class._run = patch_run(  # type: ignore[attr-defined]
        new_class._run,  # type: ignore[attr-defined]
        processor,
        skip_transpile,
        backend,
        options,
    )
    return new_class
