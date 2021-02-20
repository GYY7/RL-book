from typing import Iterator, Mapping, Tuple, TypeVar, Sequence, List
from operator import itemgetter
import numpy as np

from rl.distribution import Distribution, Constant
from rl.function_approx import FunctionApprox
from rl.iterate import iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess, Policy,
                                        MarkovDecisionProcess,
                                        StateActionMapping, FinitePolicy)
from rl.dynamic_programming import *


S = TypeVar('S')
A = TypeVar('A')

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]


def policy_iteration(
    mdp: MarkovDecisionProcess[S, A],
    γ: float,
    approx_0: FunctionApprox[S],
    non_terminal_states_distribution: Distribution[S],
    num_state_samples: int
) -> Iterator[FunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step for a random sample of the process'
    non-terminal states.

    '''
    current_approx = approx_0
    while True:
        # policy evaluation
        def update(v: FunctionApprox[S]) -> FunctionApprox[S]:
            nt_states: Sequence[S] = non_terminal_states_distribution.sample_n(
                num_state_samples
            )

            def return_(s_r: Tuple[S, float]) -> float:
                s1, r = s_r
                return r + γ * v.evaluate([s1]).item()  # should apply current_policy to get the value function instead

            return v.update(
                [(s, max(mdp.step(s, a).expectation(return_,)
                         for a in mdp.actions(s)))
                 for s in nt_states]
            )
        yield current_approx
        current_approx = update(current_approx)
        # policy improvement
        current_policy = greedy_policy_from_vf(mdp, current_approx, γ)
        # should modify greedy_policy_from_vf that takes FunctionApprox as value function





def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    matrix_method_for_mrp_eval: bool = False
) -> Iterator[Tuple[V[S], FinitePolicy[S, A]]]:
    '''Calculate the value function (V*) of the given MDP by improving
    the policy repeatedly after evaluating the value function for a policy
    '''

    def update(vf_policy: Tuple[V[S], FinitePolicy[S, A]])\
            -> Tuple[V[S], FinitePolicy[S, A]]:

        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        policy_vf: V[S] = {mrp.non_terminal_states[i]: v for i, v in
                           enumerate(mrp.get_value_function_vec(gamma))}\
            if matrix_method_for_mrp_eval else evaluate_mrp_result(mrp, gamma)
        improved_pi: FinitePolicy[S, A] = greedy_policy_from_vf(
            mdp,
            policy_vf,
            gamma
        )

        return policy_vf, improved_pi

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    pi_0: FinitePolicy[S, A] = FinitePolicy(
        {s: Choose(set(mdp.actions(s))) for s in mdp.non_terminal_states}
    )
    return iterate(update, (v_0, pi_0))

