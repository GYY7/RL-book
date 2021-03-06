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

S = TypeVar('S')
A = TypeVar('A')

def td_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        count_to_weight_func: Callable[[int], float],
        gamma: float,
        tolerance: float = 1e-8
) -> Tabular[S]:
    """
    Similar as Monte Carlo Scratch except replacing return y with R_{t+1} + gamma*V(S_{t+1}) for updates
    """
    values_map: Dict[S, float] = {}
    counts_map: Dict[S, int] = {}
    count = 0
    max_steps = round(math.log(tolerance) / math.log(gamma))
    print('max steps: ', max_steps)
    for transition in transitions:
        if count >= max_steps:
            break
        else:
            state = transition.state
            counts_map[state] = counts_map.get(state, 0) + 1
            weight: float = count_to_weight_func(counts_map.get(state, 0))
            if transition.next_state not in values_map:
                values_map[transition.next_state] = -30
            y = transition.reward + gamma * values_map[transition.next_state]
            values_map[state] = weight * y + (1 - weight) * values_map.get(state, 0.)
            count += 1
    return Tabular(values_map, counts_map, count_to_weight_func)

