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


def sarsa_control(
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
        if explore.sample():
            next_action = Choose(set(state_action[next_state])).sample()
        else:
            next_action = state_action[next_state][np.argmax([q((next_state, a))
                                                              for a in state_action[next_state]])]
        q = q.update([(state, action), reward + gamma * q((next_state, next_action))])
        state, action = next_state, next_action
        yield q


# def sarsa_control(
#         mdp: markov_decision_process.MarkovDecisionProcess[S, A],
#         transitions: Iterable[markov_decision_process.TransitionStep[S, A]],
#         approx_0: FunctionApprox[Tuple[S, A]],
#         γ: float,
#         ϵ: float = 0.1
# ) -> Iterator[FunctionApprox[Tuple[S, A]]]:
#     '''Return policies that try to maximize the reward based on the given
#     set of experiences.
#
#     Arguments:
#       transitions -- a sequence of state, action, reward, state (S, A, R, S')
#       actions -- a function returning the possible actions for a given state
#       approx_0 -- initial approximation of q function
#       γ -- discount rate (0 < γ ≤ 1)
#
#     Returns:
#       an itertor of approximations of the q function based on the
#       transitions given as input
#
#     '''
#     def step(q, transition):
#         explore = Bernoulli(ϵ)
#         qa_lst = [q((transition.next_state, a)) for a in mdp.actions(transition.next_state)]
#         if explore.sample():
#             next_reward = Choose(set(qa_lst)).sample()
#         else:
#             next_reward = max(qa_lst)
#         return q.update([
#             ((transition.state, transition.action),
#              transition.reward + γ * next_reward)
#         ])
#
#     return iterate.accumulate(transitions, step, initial=approx_0)

# def sarsa_control(
#         mdp: markov_decision_process.MarkovDecisionProcess[S, A],
#         approx_0: FunctionApprox[Tuple[S, A]],
#         initial_policy: markov_decision_process.Policy[S, A],
#         γ: float,
#         max_steps: int = 1000
# ) -> FunctionApprox[Tuple[S, A]]:
#     '''Return policies that try to maximize the reward based on the given
#     set of experiences.
#
#     Arguments:
#       transitions -- a sequence of state, action, reward, state (S, A, R, S')
#       actions -- a function returning the possible actions for a given state
#       approx_0 -- initial approximation of q function
#       γ -- discount rate (0 < γ ≤ 1)
#
#     Returns:
#       an itertor of approximations of the q function based on the
#       transitions given as input
#
#     '''
#     start_states = Categorical({state: 1/len(mdp.non_terminal_states) for state in mdp.non_terminal_states})
#     for trans in mdp.simulate_actions(start_states, initial_policy):  # Iterable[TransitionStep[S, A]]
#         transition = trans
#         break
#
#     q = approx_0
#     policy = initial_policy
#     count = 0
#     while count < max_steps:
#         count += 1
#         for next_trans in mdp.simulate_actions(Constant(transition.next_state), policy):
#             next_transition = next_trans
#             break
#         q = q.update([
#             (transition.state, transition.action), transition.reward + γ * q((transition.next_state, next_transition.action))])
#         policy = markov_decision_process.policy_from_q(q, mdp, 1 / count)
#         transition = next_transition
#         print(count, q)
#     return q


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp = SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )
    transition_map = si_mdp.get_action_transition_reward_map()
    # fdp: markov_decision_process.FinitePolicy[InventoryState, int] = markov_decision_process.FinitePolicy(
    #     {InventoryState(alpha, beta):
    #          Constant(user_capacity - (alpha + beta)) for alpha in
    #      range(user_capacity + 1) for beta in range(user_capacity + 1 - alpha)}
    # )
    # initialize values_map and count_maps for Tabular
    start_map = {}
    state_action = {}
    for state in si_mdp.mapping.keys():
        state_action[state] = []
        for action in si_mdp.actions(state):
            start_map[(state, action)] = 0
            state_action[state].append(action)

    q = Tabular(start_map, start_map)
    start_states = Categorical({state: 1/len(si_mdp.non_terminal_states) for state in si_mdp.non_terminal_states})
    # transitions = si_mdp.simulate_actions(start_states, fdp)
    
    sarsa_tabular_control = sarsa_control(start_states, transition_map, state_action, q, user_gamma, 0.1)
    diff = {}
    prev = q.values_map
    count = 0
    for fcn_approx in sarsa_tabular_control:
        next = fcn_approx.values_map
        print(fcn_approx.values_map)
        count += 1
        max_diff = max([abs(next[key] - prev[key])for key in next])
        print(max_diff)
        prev = next
        print(max_diff)
        if count > 100 and max_diff < 1e-6:
            print(count)
            break
    # sarsa_tabular_control = sarsa_control(si_mdp, q, fdp, user_gamma)
    #
    # print(sarsa_tabular_control.values_map)
    # opt_vf, opt_pi = get_optimal_policy(values_map)
    # print('opt_vf mc control: \n', opt_vf, '\nopt_pi mc control: \n', opt_pi)
