from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar)
from rl.distribution import Distribution, Categorical, Constant
from rl.function_approx import FunctionApprox, Tabular
from rl.markov_process import FiniteMarkovProcess
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import MarkovDecisionProcess, FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess, FinitePolicy
from rl.returns import returns
from dataclasses import dataclass, replace, field
import math
import operator
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, InventoryState
from rl.dynamic_programming import evaluate_mrp_result
from rl.dynamic_programming import policy_iteration_result
from rl.dynamic_programming import value_iteration_result


S = TypeVar('S')
A = TypeVar('A')

def mc_control(
        mdp: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        γ: float,
        K: int,
        tolerance: float = 1e-6
) -> FunctionApprox[Tuple[S, A]]:
    '''Evaluate an MRP using the monte carlo method, simulating K episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mrp -- the Markov Reward Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      ϵ -- the fraction of the actions where we explore rather = 1/k
      than following the optimal policy
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated Q function
    after each episode.

    '''
    q = approx_0
    p = markov_decision_process.policy_from_q(q, mdp)

    for k in range(1, K+1):
        trace: Iterable[markov_decision_process.TransitionStep[S, A]] =\
            mdp.simulate_actions(states, p)
        q = q.update(
            ((step.state, step.action), step.return_)
            for step in returns(trace, γ, tolerance)
        )

        p = markov_decision_process.policy_from_q(q, mdp, 1/k)
    return q


def get_optimal_policy(values_map: Dict[Tuple[S, A], float]):
    opt_vf = {}
    opt_pi = {}
    for key in values_map:
        state, action = key
        value = values_map[key]
        if (state in opt_vf and opt_vf[state] < value) or (state not in opt_vf):
            opt_vf[state] = value
            opt_pi[state] = action
    return opt_vf, opt_pi


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
    # initialize values_map and count_maps for Tabular
    start_map = {}
    for state in si_mdp.mapping.keys():
        for action in si_mdp.actions(state):
            start_map[(state, action)] = 0
    # start state distribution: every non-terminal state has equal probability to be the start state
    start_states = Categorical({state: 1/len(si_mdp.non_terminal_states) for state in si_mdp.non_terminal_states})

    mc_tabular_control = mc_control(si_mdp, start_states, Tabular(start_map, start_map), user_gamma, 800)
    values_map = mc_tabular_control.values_map
    opt_vf, opt_pi = get_optimal_policy(values_map)
    print('opt_vf mc control: \n', opt_vf, '\nopt_pi mc control: \n', opt_pi)

    fdp: FinitePolicy[InventoryState, int] = FinitePolicy(
        {InventoryState(alpha, beta):
             Constant(user_capacity - (alpha + beta)) for alpha in
         range(user_capacity + 1) for beta in range(user_capacity + 1 - alpha)}
    )
    implied_mrp: FiniteMarkovRewardProcess[InventoryState] = \
        si_mdp.apply_finite_policy(fdp)

    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    print(opt_vf_vi, '\n')
    print(opt_policy_vi)

    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    print(opt_vf_pi, '\n')
    print(opt_policy_pi)

