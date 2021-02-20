from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar, Any)
import math
import numpy as np


def utility_maximization(n_jobs: int, wages: Iterable[int], possibilities: Iterable[float], gamma: float, alpha: float,
                         wage_unemp: int) -> Iterable[float]:
    value_function = [0 for i in range(n_jobs) for j in range(2)]  # 2i: (Employed, i+1); 2i+1: (Unemployed, i+1)
    epsilon = 1e-5
    diff = 1
    while diff >= epsilon:
        prev_vf = [value for value in value_function]
        for i in range(n_jobs):
            expected_unemp = sum([prev_vf[2*j+1]*possibilities[j] for j in range(n_jobs)])
            v_emp = math.log(wages[i]) + gamma*alpha*expected_unemp + gamma*(1-alpha)*prev_vf[2*i]
            v_unemp = max(math.log(wage_unemp) + gamma*expected_unemp, prev_vf[2*i])
            value_function[2*i] = v_emp
            value_function[2*+1] = v_unemp
        diff = max(abs(np.array(prev_vf) - np.array(value_function)))
    return value_function


def get_optimal_policy(value_function: Iterable[float], n_jobs: int) -> Iterable[str]:
    """
    If employed, can only choose Accept; If unemployed, can choose Accept if new job offer gives higher optimal value,
    otherwise Decline.
    """
    total = len(value_function)
    # 2i: (Employed, i+1); 2i+1: (Unemployed, i+1)
    optimal_policy = ['Decline' if i % 2 == 1 and value_function[2*i] < value_function[2*i+1] else 'Accept'
                      for i in range(n_jobs)]
    return optimal_policy

