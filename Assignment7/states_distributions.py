from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian, Categorical
from typing import Sequence, Callable, Tuple, Iterator
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox


class AssetAllocDiscrete:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    risky_alloc_choices: Sequence[float]
    feature_functions: Sequence[Callable[[Tuple[float, float]], float]]
    dnn_spec: DNNSpec
    initial_wealth_distribution: Distribution[float]


def calculate_states_distribution(self, t: int) -> Sequence[Distribution[float]]:
    """
    W_{t+1} = pi_t(1+mu_t) + (W_t - pi_t)(1+r_t)
    dW_t = (pi_t(mu_t-r_t) + r_t)W_t dt
    For the state distributions, create a sequence of distributions whose i-th element
    is the cummulative distribution of wealth
    """
    result = []
    actions_distr: Choose[float] = self.uniform_actions()
    cdf_wealth: float = self.initial_wealth_distribution.sample()
    for i in range(t):
        cdf_wealth = Distribution[float]
        # integrate dW_t = (pi_t(mu_t-r_t) + r_t)W_t dt
        # given cdf_wealth(prev), risky_return_distributions, actions_distr, riskless_returns
        result.append(cdf_wealth)
    return result

def calculate_risky_prices_distribution(self, t: int) -> Sequence[Distribution[float]]:
    """
    Derive the distributions of risky asset price at any time step
    Calculate the probability of pi_t(1+mu_t)
    """
    result = []
    actions_distr: Choose[float] = self.uniform_actions()
    for i in range(t):
        risky_pdf = Distribution[float]  # the pdf of pi_t(1+mu_t) given risky_return_distributions, actions_distr
        result.append(risky_pdf)
    return result
