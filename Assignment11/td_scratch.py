from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar)
from rl.distribution import Distribution
from rl.function_approx import FunctionApprox, Tabular
import rl.markov_process as mp
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import (MarkovDecisionProcess)
from rl.returns import returns
from dataclasses import dataclass, replace, field
import math
import operator

S = TypeVar('S')
A = TypeVar('A')

def td_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        count_to_weight_func: Callable[[int], float],
        gamma: float,
        max_steps: int = 5000
) -> Tabular[S]:
    """
    Similar as Monte Carlo Scratch except replacing return y with R_{t+1} + gamma*V(S_{t+1}) for updates
    """
    values_map: Dict[S, float] = {}
    counts_map: Dict[S, int] = {}
    count = 0
    diff = {}  # dict: state and its value error
    for transition in transitions:
        if count < max_steps:
            state = transition.state
            if state not in diff:
                diff[state] = 100
            counts_map[state] = counts_map.get(state, 0) + 1
            weight: float = count_to_weight_func(counts_map.get(state, 0))
            if transition.next_state not in values_map:
                values_map[transition.next_state] = -30

            y = transition.reward + gamma * values_map[transition.next_state]
            diff[state] = min(abs(y - values_map.get(state, 0.)), diff[state])
            values_map[state] = weight * y + (1 - weight) * values_map.get(state, 0.)
            count += 1
        elif count >= max_steps or diff[max(diff.items(), key=operator.itemgetter(1))[0]] < 1e-4:
            print(diff[max(diff.items(), key=operator.itemgetter(1))[0]])
            break

    return Tabular(values_map, counts_map, count_to_weight_func)

