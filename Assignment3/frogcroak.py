from rl import *
from rl.markov_decision_process import FiniteMarkovDecisionProcess, StateActionMapping, FinitePolicy, FiniteMarkovRewardProcess
from dataclasses import dataclass
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar, Any)
from rl.distribution import Categorical, Constant
import itertools
import numpy as np
import matplotlib.pyplot as plt

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
                prob_dict_b: Dict[Tuple[FrogState, float], float] = {(FrogState(j), 1/(self.num_pads-1)): 1/(self.num_pads-1)
                                                                     for j in range(self.num_pads)}
                d1['B'] = Categorical(prob_dict_b)
                d[state] = d1
        return d



def main(num_pads):
    # 2^(num_pads-2) deterministic policies
    fc_mdp: FiniteMarkovDecisionProcess[FrogState, Any] = FrogCroak(num_pads+1)
    all_fp = list(itertools.product(['A', 'B'], repeat=fc_mdp.num_pads-2))
    all_mrp_value = []
    for fp in all_fp:
        fdp: FinitePolicy[FrogState, Any] = FinitePolicy({FrogState(i+1): Constant(fp[i]) for i in range(len(fp))})
        implied_mrp: FiniteMarkovRewardProcess[FrogState] = fc_mdp.apply_finite_policy(fdp)
        all_mrp_value.append(implied_mrp.get_value_function_vec(1))

    # find the optimized policy
    max_indices = []
    value_matrix = np.array(all_mrp_value)
    for i in range(num_pads-1):
        max_indices.append(np.argmax(value_matrix[:, i]))
    max_index = list(set(max_indices))[0]
    print(value_matrix[max_index, :])
    print(all_fp[max_index])
    plt.plot(['State'+str(i+1)+','+ all_fp[max_index][i] for i in range(num_pads-1)], value_matrix[max_index, :],'o')
    plt.xlabel('Frog State')
    plt.ylabel('Probability')
    plt.title('n = ' + str(num_pads-1))
    plt.show()

if __name__ == '__main__':
    main(4)
    main(7)
    main(10)
