import numpy as np
from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar)
from rl.distribution import Distribution, Categorical, Constant
from rl.function_approx import FunctionApprox
from rl.markov_process import FiniteMarkovProcess
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import MarkovDecisionProcess, FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess, FinitePolicy
from rl.returns import returns
from dataclasses import dataclass, replace, field

S = TypeVar('S')
A = TypeVar('A')


def mc_policy_gradient(episodes, m: int, n: int, T: int,
                       start_distribution: Distribution[Tuple[S, A]], policy_approx: FunctionApprox[S, A],
                       generate_trace_fcn, gamma:float, alpha: float, calculate_gradient: FunctionApprox):
    """
    MC Policy Gradient
    """
    theta = np.random.rand(m, 1)
    for i in range(n): # n episodes
        start_state, start_action = start_distribution.sample()
        episode = generate_trace_fcn(start_state, start_action, policy_approx)
        for t in range(T):
            state, action = episode[3*t], episode[3*t+1]
            # update G and theta
            G = sum([gamma**(k-t)*episode[3*k+2] for k in range(t, T+1)]) # R_k+1: episode[3*k+2]
            theta += alpha* gamma**t * calculate_gradient(policy_approx(state, action)) * G
            policy_approx.update(theta)
    return policy_approx


