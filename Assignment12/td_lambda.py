from typing import Callable, Iterable, Iterator, TypeVar, Tuple, Dict

from rl.function_approx import FunctionApprox, Tabular
import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.iterate as iterate
from rl.returns import returns
import operator

S = TypeVar('S')

def td_lambda_tabular_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        count_to_weight_func: Callable[[int], float],
        gamma: float,
        lambd: float,
        max_steps: int = 2000,
        tolerance: float = 1e-200
) -> Tuple[Tabular[S], int]:
    """
    Similar to TD Scratch except replacing use G_{t,n} for updates
    """
    values_map: Dict[S, float] = {}
    counts_map: Dict[S, int] = {}
    trace = []
    count = 0
    diff = {}  # dict: state and its value error
    for transition in transitions:
        count += 1
        trace.append(transition)
        if count > max_steps:
            break

    # get corresponding return
    transitions_returns = returns(trace, gamma, tolerance)
    trace_returns = [return_ for return_ in transitions_returns]

    for i in range(max_steps):
        transition = trace[i]
        state = transition.state
        if state not in diff:
            diff[state] = 100
        counts_map[state] = counts_map.get(state, 0) + 1
        weight: float = count_to_weight_func(counts_map.get(state, 0))
        if transition.next_state not in values_map:
            values_map[transition.next_state] = -30
        y = lambd**(max_steps-i-1) * trace_returns[i].return_
        if lambd == 0:
            y = 0
        for n in range(1, max_steps-i):
            g_tn = 0
            for j in range(i, i+n):
                next_transition = trace[j]
                g_tn += gamma**(j-i) * next_transition.reward
                if j == i+n-1:
                    g_tn += gamma**n * values_map.get(next_transition.next_state, 0)
            y += (1-lambd) * lambd**(n-1) * g_tn
        diff[state] = min(abs(y - values_map.get(state, 0.)), diff[state])
        values_map[state] = weight * y + (1 - weight) * values_map.get(state, 0.)
        # print(y, values_map[state])
        count += 1
        if diff[max(diff.items(), key=operator.itemgetter(1))[0]] < 0.1:
            break
    print(diff[max(diff.items(), key=operator.itemgetter(1))[0]])
    return Tabular(values_map, counts_map, count_to_weight_func), i
