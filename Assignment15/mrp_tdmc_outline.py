from typing import Sequence, Tuple, Mapping, Dict
import numpy as np
from rl.function_approx import LinearFunctionApprox, Weights

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    values_map: Dict[S, float] = {}
    counts_map: Dict[S, int] = {}
    for state, return_ in state_return_samples:
        counts_map[state] = counts_map.get(state, 0) + 1
        weight: float = 1/counts_map.get(state, 0)
        values_map[state] = weight * return_ + (1 - weight) * values_map.get(state, 0.)
    return values_map


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    prob_func = {}
    state_counts = {}
    reward_func = {}
    for state, reward, next_state in srs_samples:
        if state not in prob_func:
            prob_func[state] = {}
        if next_state not in prob_func[state]:
            prob_func[state][next_state] = 0
        reward_func[state] = reward_func.get(state, 0) + reward
        state_counts[state] = state_counts.get(state, 0) + 1
        prob_func[state][next_state] += 1
    for state, count in state_counts.items():
        reward_func[state] /= count
        for next_state in prob_func[state]:
            prob_func[state][next_state] /= count
    return prob_func, reward_func


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    values_map: Dict[S, float] = {}
    for update in range(50):
        for state, reward in reward_func.items():
            values_map[state] = reward_func[state] + sum([prob*values_map.get(next_state, 0)
                                                          for next_state, prob in prob_func[state].items()])
    return values_map


def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 30000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    values_map: Dict[S, float] = {}
    counts_map: Dict[S, int] = {}
    for update in range(1, num_updates+1):
        for state, reward, next_state in srs_samples:
            y = reward + values_map.get(next_state, 0.)
            counts_map[state] = counts_map.get(state, 0) + 1
            # weight: float = learning_rate*(counts_map[state]/learning_rate_decay + 1) ** (-0.5)
            weight = learning_rate*(update/learning_rate_decay + 1) ** (-0.5)
            values_map[state] = weight * y + (1 - weight) * values_map.get(state, 0.)
    return values_map

def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    # all_states = list(set([state for state, reward, next_state in srs_samples]))
    # weight = Weights.create(np.zeros((len(all_states), 1)))
    # approx = LinearFunctionApprox([lambda x: x == state for state in all_states], 0, weight, False)
    # how to use  LinearFunctionApprox.update?
    all_states = list(set([state for state, reward, next_state in srs_samples] +
                          [next_state for state, reward, next_state in srs_samples]))
    m = len(all_states)
    A = np.random.rand(m, m)
    b = np.zeros((m, 1))
    feature_map = np.identity(m)
    w = np.zeros((m, 1))

    for state, reward, next_state in srs_samples:
        state_idx = all_states.index(state)
        next_state_idx = all_states.index(next_state)
        feature_state = feature_map[state_idx].reshape((m, 1))
        feature_next_state = feature_map[next_state_idx].reshape((m, 1))
        A += feature_state @ (feature_state - feature_next_state).T
        b += feature_state * reward
        w = np.linalg.inv(A) @ b
    return {all_states[i]: np.array(feature_map[i]) @ w for i in range(m)}


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))
