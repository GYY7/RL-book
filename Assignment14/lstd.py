from typing import Callable, Iterable, Iterator, TypeVar, Tuple, Dict, List
import rl.markov_process as mp
import numpy as np

S = TypeVar('S')


def lstd(transitions: Iterable[mp.TransitionStep[S]],
         feature_map: Dict[S, List[float]],
         m: int,
         gamma: float) -> Iterable[Dict[S, float]]:
    """
    update A and b to get w*= inverse(A)b and value functions
    feature_map:  key: state, value: phi(s_i) is a vector of dimension m
    """
    # initialize A, b
    A = np.random.rand(m, m)
    b = np.zeros((m, 1))
    for transition in transitions:
        feature_state = np.array(feature_map[transition.state])
        feature_next_state = np.array(feature_map[transition.next_state])
        A += feature_state @ (feature_state - gamma*feature_next_state).T
        b += feature_state * transition.reward
        w = np.linalg.inv(A) @ b
        yield {state: np.array(feature_map[state]) @ w for state in feature_map.keys()}
