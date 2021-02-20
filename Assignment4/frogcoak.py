from rl import *
from rl.markov_decision_process import FiniteMarkovDecisionProcess, StateActionMapping, FinitePolicy, FiniteMarkovRewardProcess
from dataclasses import dataclass
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar, Any)
from rl.distribution import Categorical, Constant
import itertools
import numpy as np
import matplotlib.pyplot as plt
from rl.dynamic_programming import *
import time

@dataclass(frozen=True)
class FrogState:
    """
    Frog state
    """
    position: int


FrogCroakMapping = StateActionMapping[FrogState, float]


class FrogCroak(FiniteMarkovDecisionProcess[FrogState, float]):

    def __init__(self, num_pads):
        self.num_pads = num_pads
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> FrogCroakMapping:
        d: Dict[FrogState, Dict[Any, Categorical[Tuple[FrogState, float]]]] = {}

        for i in range(self.num_pads):
            state = FrogState(i)
            if i == 0 or i == self.num_pads - 1:
                d[state] = None
            else:
                d1: Dict[Any, Categorical[Tuple[FrogState, float]]] = {}
                _reward = 0
                if i == self.num_pads - 2:
                    _reward = 1 - i / (self.num_pads - 1)
                prob_dict_a: Dict[Tuple[FrogState, float], float] = {(FrogState(i + 1), 0): 1 - i / (self.num_pads - 1),
                        (FrogState(i - 1), _reward): i / (self.num_pads - 1)}
                d1['A'] = Categorical(prob_dict_a)
                prob_dict_b: Dict[Tuple[FrogState, float], float] = {(FrogState(j), 0): 1/(self.num_pads-1)
                                                                     for j in range(self.num_pads) if j != i or
                                                                     j != self.num_pads - 2}
                prob_dict_b[(FrogState(self.num_pads-2), 1/(self.num_pads-1))] = 1/(self.num_pads-1)
                d1['B'] = Categorical(prob_dict_b)
                d[state] = d1
        return d

if __name__ == '__main__':
    speed_v = []
    speed_p = []
    for num_pads in range(4, 12):
        mdp = FrogCroak(num_pads)
        t1 = time.time()
        opt_vfv, opt_pov = value_iteration_result(mdp, 1)
        t2 = time.time()
        opt_vfp, opt_pop = policy_iteration_result(mdp, 1)
        t3 = time.time()
        speed_v.append(t2-t1)
        speed_p.append(t3-t2)
    plt.plot(np.arange(4, 12), speed_v, 'o' )
    plt.plot(np.arange(4, 12), speed_p, 'o')
    plt.legend(['value iterations', 'policy iterations'])
    plt.ylabel('Time to converge')
    plt.xlabel('Number of pads')
    plt.show()

