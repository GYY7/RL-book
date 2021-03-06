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

# returns(trace, γ, tolerance):
def mc_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        count_to_weight_func: Callable[[int], float],
        gamma: float,
        tolerance: float = 1e-200
) -> Tabular[S]:
    '''
    Returns the approximated value
    function after each episode.

    Approximates Tabular MC Prediction with a discrete domain of states S, without any
    interpolation. The value function for each S is maintained as a weighted
    mean of observations by recency (managed by
    `count_to_weight_func').

    In practice, this means you can use this to approximate a function
    with a learning rate α(n) specified by count_to_weight_func.


    Fields:
    values_map -- mapping from S to its approximated value function
    counts_map -- how many times a given S has been updated
    count_to_weight_func -- function for how much to weigh an update
      to S based on the number of times that S has been updated

    Update the value approximation with the given points.
    '''
    values_map: Dict[S, float] = {}
    counts_map: Dict[S, int] = {}
    trace = []
    count = 0
    diff = {}
    max_steps = round(math.log(tolerance) / math.log(gamma))
    print('max steps: ', max_steps)
    # get trace
    for transition in transitions:
        trace.append(transition)
        count += 1
        if count >= max_steps:
            break
    # get corresponding return
    transitions_returns = returns(trace, gamma, tolerance)
    trace_returns = [return_ for return_ in transitions_returns]

    for i in range(len(trace)):
        # x: state; y: return for first n occurrences of x
        x = trace[i].state
        y = trace_returns[i].return_
        if x not in diff:
            diff[x] = 100
        diff[x] = min(abs(y - values_map.get(x, 0.)), diff[x])
        if diff[max(diff.items(), key=operator.itemgetter(1))[0]] < 1e-4:
            break
        counts_map[x] = counts_map.get(x, 0) + 1
        weight: float = count_to_weight_func(counts_map.get(x, 0))
        values_map[x] = weight * y + (1 - weight) * values_map.get(x, 0.)
    print(diff[max(diff.items(), key=operator.itemgetter(1))[0]])
    return Tabular(values_map, counts_map, count_to_weight_func)



