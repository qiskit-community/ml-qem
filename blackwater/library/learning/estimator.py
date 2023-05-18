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
# from .mlp import encode_data
from mlp import encode_data
import scipy
from qiskit.transpiler.exceptions import TranspilerError


class LearningMethodEstimatorProcessor:
    def process(
            self,
            expectation_value: np.ndarray[Any, np.dtype[np.float64]],
            circuits: Union[QuantumCircuit, List[QuantumCircuit]],
            observables: Union[PauliSumOp, List[PauliSumOp]],
            parameter_values: Tuple[Tuple[float, ...], ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        raise NotImplementedError


class ZNEProcessor(LearningMethodEstimatorProcessor):
    def __init__(self,
                 zne_estimator,
                 zne_strategy,
                 backend: BackendV1,
                 shots=10000):
        self._zne_estimator = zne_estimator
        self._zne_strategy = zne_strategy
        self._backend = backend
        self._shots = shots

    def process(
            self,
            expectation_value: np.ndarray[Any, np.dtype[np.float64]],
            circuits: Union[QuantumCircuit, List[QuantumCircuit]],
            observables: Union[PauliSumOp, List[PauliSumOp]],
            parameter_values: Tuple[Tuple[float, ...], ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:

        circuits_with_meas = circuits.copy()
        circuits_with_meas.measure_all()

        success = False
        while not success:
            try:
                circuits = transpile(circuits, backend=self._backend, optimization_level=0)
                circuits_with_meas = transpile(circuits_with_meas, backend=self._backend, optimization_level=0)
                success = True
            except (scipy.linalg.LinAlgError, TranspilerError, np.linalg.LinAlgError) as e:
                print(f"Ran into an error:, {e}")

        def form_all_qubit_observable(observable, measurement_qubits, total_num_qubits):
            assert len(observable) == len(measurement_qubits)
            converted_obs = list('I' * total_num_qubits)
            for qubit, basis in zip(measurement_qubits, list(observable)):
                converted_obs[qubit] = basis
            return ''.join(converted_obs)[::-1]

        def get_measurement_qubits(qc, num_measured_qubit):
            measurement_qubits = []
            for measurement in range(num_measured_qubit - 1, -1, -1):
                measurement_qubits.append(qc.data[-1 - measurement][1][0].index)
            return measurement_qubits

        converted_obs = []
        for str_pauli, coeff in observables.to_list():
            padded_str_pauli = form_all_qubit_observable(str_pauli[::-1], get_measurement_qubits(circuits_with_meas, 2), 5)
            converted_obs.append((padded_str_pauli, coeff))
        converted_obsservables = SparsePauliOp.from_list(converted_obs)

        job = self._zne_estimator.run(circuits, converted_obsservables, shots=self._shots, zne_strategy=self._zne_strategy)
        values = job.result().values[0]

        return values



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

        num_qubits = circuits.num_qubits

        success = False
        while not success:
            try:
                circuits = transpile(circuits, backend=self._backend, optimization_level=0)
                success = True
            except (scipy.linalg.LinAlgError, TranspilerError, np.linalg.LinAlgError) as e:
                print(f"Ran into an error:, {e}")

        results = []
        for p in observables:
            coeff = p.coeffs
            pauli = p.paulis

            # pauli_non_endian = pauli[0][::-1]
            # meas_circ = QuantumCircuit(num_qubits)
            # for i in range(num_qubits):
            #     if str(pauli_non_endian[i]) in 'IZ':
            #         pass
            #     elif str(pauli_non_endian[i]) == 'X':
            #         meas_circ.h(i)
            #     elif str(pauli_non_endian[i]) == 'Y':
            #         meas_circ.sdg(i)
            #         meas_circ.h(i)
            # circuits.compose(meas_circ, inplace=True)


            model_input, _ = encode_data(
                circuits=[circuits],
                properties=self._properties,
                ideal_exp_vals=[[0.]],
                noisy_exp_vals=[[expectation_value]],
                num_qubits=1,
                meas_bases=encode_pauli_sum_op(SparsePauliOp(pauli))
            )

            output = self._model.predict(model_input).item()

            results.append(output * np.real(coeff[0]))
            # results.append(output)

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
                    circuit, self.backend(), optimization_level=3, **self._options.__dict__
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