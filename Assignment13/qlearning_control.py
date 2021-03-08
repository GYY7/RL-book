from typing import Callable, Iterable, Iterator, TypeVar, Tuple, Mapping, List

from rl.function_approx import FunctionApprox, Tabular
import rl.markov_process as mp
import rl.markov_decision_process as markov_decision_process
import rl.iterate as iterate
from rl.distribution import Bernoulli, Choose, Categorical, Constant, Distribution
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, InventoryState
import numpy as np
import operator

S = TypeVar('S')
A = TypeVar('A')

# Callable[[S], Optional[Iterable[A]]]


def qlearning_control(
        start_states: Distribution[S],
        transition_fcn: Callable[[S, A], Tuple[S, float]],
        state_action: Mapping[S, List[A]],
        approx_0: FunctionApprox[Tuple[S, A]],
        gamma: float,
        ϵ: float
) -> Iterable[FunctionApprox[Tuple[S, A]]]:
    """
    Update Q-value function approximate using SARSA
    Initialize first state by start_states
    """
    q = approx_0
    state = start_states.sample()
    action = Choose(set(state_action[state])).sample()
    while True:
        # next_state, reward = transition_fcn(state, action)
        next_state, reward = transition_fcn[state][action].sample()
        # use ϵ-greedy policy to get next_action
        explore = Bernoulli(ϵ)
        qa_list = [q((next_state, a)) for a in state_action[next_state]]
        if explore.sample():
            next_action = Choose(set(state_action[next_state])).sample()
        else:
            next_action = state_action[next_state][np.argmax(qa_list)]
        # update q with max{Q(S_{t+1}, a'} a' in action space
        q = q.update([(state, action), reward + gamma * max(qa_list)])
        state, action = next_state, next_action
        yield q


