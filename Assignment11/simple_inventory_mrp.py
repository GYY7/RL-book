from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite, InventoryState
from rl.markov_process import FiniteMarkovProcess
from rl.distribution import Categorical
from Assignment11.td_scratch import td_prediction
from dataclasses import dataclass, replace, field
from Assignment11.monte_carlo_scratch import mc_prediction

user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = 1.0
user_stockout_cost = 10.0

user_gamma = 0.9

si_mrp = SimpleInventoryMRPFinite(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda,
    holding_cost=user_holding_cost,
    stockout_cost=user_stockout_cost
)

print("-----MRP Value Function-----:\n")
print(si_mrp.get_value_function_vec(user_gamma))


# create Iterable[TransitionStep[S]]
non_terminal_states = si_mrp.non_terminal_states
start_distribution = {state: 1/len(non_terminal_states) for state in non_terminal_states}
transitions = si_mrp.simulate_reward(Categorical(start_distribution))

def count_to_weight_func(n: int):
    return 1/n

print("-----TD Value Function-----:\n")
td_pred = td_prediction(transitions, count_to_weight_func, user_gamma)
print(td_pred.evaluate(non_terminal_states))

print("-----MC Value Function-----:\n")
mc_pred = mc_prediction(transitions, count_to_weight_func, user_gamma)
print(mc_pred.evaluate(non_terminal_states))
