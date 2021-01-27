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
    """
    Stock price state
    """
    price: int


@dataclass
class StockPriceMRP1(MarkovRewardProcess[StateMP1]):
    """
    Stock price Markov reward process in Chapter 1
    """
    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def __init__(self, reward_function, level_param):
        self.reward_function = reward_function
        self.level_param = level_param

    def up_prob(self, state: StateMP1) -> float:
        """
        return the probability that the stock price goes up
        """
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition_reward(self, state: StateMP1) -> Optional[Distribution[Tuple[StateMP1, float]]]:
        """
        return
        """
        up_p = self.up_prob(state)

        return Categorical({
            (StateMP1(state.price + 1), self.reward_function(state.price)): up_p,
            (StateMP1(state.price - 1), self.reward_function(state.price)): 1 - up_p
        })

    def get_value_function(self, state: StateMP1, gamma: float, time: int) -> float:
        """
        return the value function of the state recursively;
        Since the process is infinite-states and non-terminating, approximate by calculating the next 20 steps
        """
        if time == 20:
            return self.reward_function(state.price)
        else:
            value_fcn = self.reward_function(state.price)
            transition_map = self.transition_reward(state)
            for key, proba in transition_map.table().items():
                next_state = key[0]
                value_fcn += gamma*proba*self.get_value_function(next_state, gamma, time + 1)
            return value_fcn


if __name__ == '__main__':

    def f(x):
        """
        customize reward function
        """
        return x

    s_mrp = StockPriceMRP1(f, 10)
    print('The value function of stock price at 5:', s_mrp.get_value_function(StateMP1(5), 0.3, 0))
