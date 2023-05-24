"""Learning estimator."""

from functools import wraps
from typing import Callable, Tuple, List, Union, Type, Optional, Any

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.providers import JobV1 as Job, Options, BackendV2
from qiskit.quantum_info import SparsePauliOp

from blackwater.data.core import DataEncoder
from blackwater.exception import BlackwaterException


class BlackWaterEstimatorModel:
    """Base model for learning based estimator primitive."""

    def __init__(self, encoder: DataEncoder):
        """Constructor for BlackWaterEstimatorModel

        Args:
            encoder: input data encoder for model
        """
        self.encoder = encoder

    def predict(
        self,
        exp_val: float,
        circuit: QuantumCircuit,
        operator: PauliSumOp,
        backend: Optional[BackendV2] = None,
    ) -> float:
        """

        Args:
            exp_val: noisy expectation value from estimator
            circuit: circuit from estimator
            operator: operator from estimator
            backend: backend from estimator

        Returns:
            mitigated expectation value
        """
        return self.run(
            self.encoder.encode(
                exp_val=exp_val, circuit=circuit, operator=operator, backend=backend
            )
        )

    def run(self, encoded_inputs: Any) -> float:
        """Executes model on encoded data.

        Args:
            encoded_inputs: inputs to model from encoder

        Returns:
            prediction a.k.a mitigated exp value
        """
        raise NotImplementedError


class ScikitLearnEstimatorModel(BlackWaterEstimatorModel):
    """Learning based model for scikit learn."""

    def __init__(self, model, encoder: DataEncoder):
        """Constructor for ScikitLearnEstimatorModel

        Args:
            model: scikit model
            encoder: numpy data encoder
        """
        super().__init__(encoder)
        self.model = model

    def run(self, encoded_inputs: Any):
        return self.model.predict([encoded_inputs]).item()


class TorchGeometricEstimatorModel(BlackWaterEstimatorModel):
    """Learning based model for pytorch geometric."""

    def __init__(self, model: torch.nn.Module, encoder: DataEncoder):
        """Constructor for ScikitLearnEstimatorModel

        Args:
            model: torch model
            encoder: torch geometric data encoder
        """
        super().__init__(encoder)
        self.model = model

    def run(self, encoded_inputs: Any):
        data, noisy_exp_val = encoded_inputs
        noisy_exp_val, _ = torch.tensor_split(  # pylint: disable=no-member
            data.y, 2, dim=1
        )
        mitigated_value = self.model(
            data.x,
            data.edge_index,
            noisy_exp_val,
            data.observable,
            data.circuit_depth,
            data.batch,
        ).item()
        return mitigated_value


class LearningEstimatorJob(Job):
    """Learning estimator wrapper for job results."""

    def __init__(  # pylint: disable=super-init-not-called
        self,
        base_job: Job,
        model: BlackWaterEstimatorModel,
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
        metadata = []
        for value, circuit, obs, params, meta in zip(
            result.values,
            self._circuits,
            self._observables,
            self._parameter_values,
            result.metadata,
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

            mitigated_value = self._model.predict(
                exp_val=value,
                circuit=bound_circuit,
                operator=obs,
                backend=self._backend,
            )

            mitigated_values.append(mitigated_value)
            metadata.append({**meta, **{"original_value": value}})

        return EstimatorResult(np.array(mitigated_values), metadata)

    def submit(self):
        return self._base_job.submit()

    def status(self):
        return self._base_job.status()

    def cancel(self):
        return self._base_job.cancel()

    def __repr__(self):
        return f"<LearningEstimatorJob: {self._base_job.job_id()}>"


def patch_run(
    run: Callable,
    model: BlackWaterEstimatorModel,
    backend: BackendV2,
    skip_transpile: bool,
    options: Optional[Options] = None,
) -> Callable:
    """Wraps run with NGEM mitigation."""

    @wraps(run)
    def learning_estimator_run(
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
        return LearningEstimatorJob(
            job,
            model=model,
            backend=backend,
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            skip_transpile=skip_transpile,
            options=options,
        )

    return learning_estimator_run


def learning_estimator(
    cls: Type[BaseEstimator],
    model: BlackWaterEstimatorModel,
    backend: BackendV2,
    skip_transpile: bool = False,
    options: Optional[Options] = None,
):
    """Decorator to turn Estimator into learning estimator.

    Args:
        cls: estimator
        model: model
        backend: backend
        skip_transpile: skip transpilation
        options: options

    Returns:
        learning estimator class
    """
    new_class: type = type(f"LearningEstimator{cls.__name__}", (cls,), {})
    new_class._run = patch_run(  # type: ignore[attr-defined]
        new_class._run,  # type: ignore[attr-defined]
        model,
        backend,
        skip_transpile,
        options,
    )
    return new_class
