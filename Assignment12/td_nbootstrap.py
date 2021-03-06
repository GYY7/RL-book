from typing import Callable, Iterable, Iterator, TypeVar, Tuple, Dict

from rl.function_approx import FunctionApprox, Tabular
import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.iterate as iterate
import operator

S = TypeVar('S')

def td_nbootstrap_tabular_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        count_to_weight_func: Callable[[int], float],
        gamma: float,
        n: int,
        max_steps: int = 5000,
        tolerance: float = 1e-10
) -> Tabular[S]:
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
        if count > max_steps+n:
            break
    for i in range(max_steps):
        transition = trace[i]
        state = transition.state
        if state not in diff:
            diff[state] = 100
        counts_map[state] = counts_map.get(state, 0) + 1
        weight: float = count_to_weight_func(counts_map.get(state, 0))
        if transition.next_state not in values_map:
            values_map[transition.next_state] = -10
        y = transition.reward
        for j in range(i+1, i+n):
             next_transition = trace[j]
             y += gamma**(j-i) * next_transition.reward
             if j == i+n-1:
                y += gamma**n * values_map.get(next_transition.next_state, 0)
        diff[state] = min(abs(y - values_map.get(state, 0.)), diff[state])
        values_map[state] = weight * y + (1 - weight) * values_map.get(state, 0.)
        count += 1
        if diff[max(diff.items(), key=operator.itemgetter(1))[0]] < 1e-4:
            break
    print(diff[max(diff.items(), key=operator.itemgetter(1))[0]])
    return Tabular(values_map, counts_map, count_to_weight_func)


def td_nbootstrap_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        γ: float,
        n: int
) -> Iterator[FunctionApprox[S]]:
    '''Evaluate an MRP using TD(0) using the given sequence of
    transitions.

    Each value this function yields represents the approximated value
    function for the MRP after an additional transition.

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)

    '''
    def step(v, transition):
        return v.update([(transition.state,
                          transition.reward + γ * v(transition.next_state))])

    # how to replace transition.reward + γ * v(transition.next_state) with G_{t,n}
    return iterate.accumulate(transitions, step, initial=approx_0)

