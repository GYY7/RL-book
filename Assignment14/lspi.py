from typing import Callable, Iterable, Iterator, TypeVar, Tuple, Dict, List
import rl.markov_decision_process as mdp
from rl.distribution import Categorical, Choose, Bernoulli
import numpy as np
import random

S = TypeVar('S')
A = TypeVar('A')


def lspi(memory: List[mdp.TransitionStep[S]],
         feature_map: Dict[Tuple[S, A], List[float]],
         state_action: Dict[S, List[A]],
         m: int,
         gamma: float,
         ϵ: float) -> Iterable[Dict[Tuple[S, A], float]]:
    """
    update A and b to get w*= inverse(A)b and update deterministic policy
    feature_map:  key: state, value: phi(s_i) is a vector of dimension m
    """
    # initialize A, b
    A = np.random.rand(m, m)
    b = np.zeros((m, 1))
    w = np.linalg.inv(A) @ b
    while True:
        transition = random.choice(memory)
        state = transition.state
        next_state = transition.next_state
        feature_state = np.array(feature_map[(state, transition.action)])
        # next_action is derived from ϵ-policy
        explore = Bernoulli(ϵ)
        if explore.sample():
            next_action = Choose(set(state_action[next_state])).sample()
        else:
            next_action = state_action[next_state][np.argmax(
                [np.array(feature_map[(next_state, action)]) @ w
                 for action in state_action[next_state]])]
        feature_next_state = np.array(feature_map[(next_state, next_action)])
        A += feature_state @ (feature_state - gamma*feature_next_state).T
        b += feature_state * transition.reward
        w = np.linalg.inv(A) @ b
        yield {s_a: np.array(feature_map[s_a]) @ w for s_a in feature_map.keys()}


