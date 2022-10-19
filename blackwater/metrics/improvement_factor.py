"""Improvement factor.

https://arxiv.org/pdf/2210.07194.pdf
"""
import typing
from dataclasses import dataclass
from math import sqrt
from typing import Optional, List, Union, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from blackwater.exception import BlackwaterException


@dataclass
class Trial:
    """Trial.

    Args:
        noisy: noisy exp value
        mitigated: exp value after mitigation
    """

    noisy: float
    mitigated: float


@dataclass
class Problem:
    """Problem.

    Args:
        trials: list of trials
        ideal_exp_value: true exp value
        circuit: circuit of a problem
        observable: observable of a problem
    """

    trials: List[Trial]
    ideal_exp_value: float
    circuit: Optional[QuantumCircuit] = None
    observable: Optional[Operator] = None


@typing.no_type_check
def improvement_factor(
    problems: Union[List[Problem], List[Tuple[float, List[Tuple[float, float]]]]],
    n_shots: int,
    n_mitigation_shots: int,
):
    """Calculates improvement factor.

    @see https://arxiv.org/pdf/2210.07194.pdf

    Example:
        >>> factor = improvement_factor(
        >>>     problems=[
        >>>         Problem(
        >>>             trials=[Trial(noisy=1.0, mitigated=2.0)],
        >>>             ideal_exp_value=0.0
        >>>         )
        >>>     ],
        >>>     n_shots=1,
        >>>     n_mitigation_shots=1,
        >>> )

    Args:
        problems: list of circuit/observable pairs and associated trials
        n_shots: shots used in evaluating noisy circuit trial
        n_mitigation_shots: shots used in evaluating mitigated circuit trial

    Returns:
        improvement factor
    """

    if len(problems) == 0:
        raise BlackwaterException("Problem list should not be empty.")

    if not isinstance(problems[0], Problem):
        problems: List[Problem] = [
            Problem(
                trials=[
                    Trial(noisy=noisy, mitigated=mitigated)
                    for noisy, mitigated in trials
                ],
                ideal_exp_value=ideal_exp_value,
            )
            for ideal_exp_value, trials in problems
        ]

    numerator = sqrt(
        n_shots
        * sum(
            sum(
                pow(trial.noisy - problem.ideal_exp_value, 2)
                for trial in problem.trials
            )
            for problem in problems
        )
    )

    denominator = sqrt(
        n_mitigation_shots
        * sum(
            sum(
                pow(trial.mitigated - problem.ideal_exp_value, 2)
                for trial in problem.trials
            )
            for problem in problems
        )
    )

    return numerator / denominator
