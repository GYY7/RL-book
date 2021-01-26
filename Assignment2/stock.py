from rl.markov_process import *
from dataclasses import dataclass
from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
import itertools
from rl.distribution import Categorical, Constant
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.chapter2.stock_price_simulations import plot_single_trace_all_processes
from rl.chapter2.stock_price_simulations import plot_distribution_at_time_all_processes


@dataclass(frozen=True)
class StateMP1:
    price: int


@dataclass
class StockPriceMP1(MarkovRewardProcess[StateMP1]):

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def __init__(self, reward_function):
        self.reward_function = reward_function

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition_reward(self, state: StateMP1) -> Optional[Distribution[Tuple[StateMP1, float]]]:
        up_p = self.up_prob(state)

        return Categorical({
            (StateMP1(state.price + 1), self.reward_function(state.price)): up_p,
            (StateMP1(state.price - 1), self.reward_function(state.price)): 1 - up_p
        })

