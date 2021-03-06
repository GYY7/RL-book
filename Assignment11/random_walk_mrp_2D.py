from typing import Mapping, Dict, Optional, Tuple
from rl.distribution import Categorical
from rl.markov_process import FiniteMarkovRewardProcess


class RandomWalk2DMRP(FiniteMarkovRewardProcess[int]):
    '''
    This MRP's states are {0, 1, 2,...,self.barrier1} * {0, 1, 2,...,self.barrier2}
    with (0,j), (self.barrier1, j), (i, 0), (i, self.barrier2) for all i, j as the terminal states.
    At each time step, we go from state (i, j) to state
    (i+1, j) with probability self.pd or to state (i-1, j) with
    probability self.pu or to state (i, j+1) with
    probability self.pr to state (i, j-1) with
    probability self.pl for all 0 < i < self.barrier1, all 0 < j < self.barrier2
    The reward is 0 if we transition to a non-terminal
    state or to terminal states (i, 0), (0, j) , and the reward is 1
    if we transition to terminal states (i, self.barrier2), (self.barrier1, j)
    '''
    barrier1: int
    barrier2: int
    pu: float  # up
    pd: float  # down
    pl: float  # left
    pr: float  # right
    def __init__(
        self,
        barrier1: int,
        barrier2: int,
        pu: float,
        pd: float,
        pl: float
    ):
        self.barrier1 = barrier1
        self.barrier2 = barrier2
        self.pu = pu
        self.pd = pd
        self.pl = pl
        self.pr = 1 - pu - pd - pl
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[Tuple[int, int], Optional[Categorical[Tuple[Tuple[int, int], float]]]]:
        d: Dict[Tuple[int, int], Optional[Categorical[Tuple[Tuple[int, int], float]]]] = {
            (i, j): Categorical({
                ((i + 1, j), 0. if i < self.barrier1 - 1 else 1.): self.pd,
                ((i - 1, j), 0.): self.pu,
                ((i, j + 1), 0. if j < self.barrier2 - 1 else 1.): self.pr,
                ((i, j - 1), 0.): self.pl,
            }) for i in range(1, self.barrier1) for j in range(1, self.barrier2)
        }
        for i in range(self.barrier1):
            d[(i, 0)] = None
            d[(i, self.barrier2)] = None
        for j in range(self.barrier2):
            d[(0, j)] = None
            d[(self.barrier1, j)] = None
        return d


if __name__ == '__main__':
    from rl.chapter10.prediction_utils import compare_td_and_mc

    this_barrier1: int = 10
    this_barrier2: int = 10
    this_pu: float = 0.25
    this_pd: float = 0.25
    this_pl: float = 0.25
    random_walk: RandomWalk2DMRP = RandomWalk2DMRP(
        barrier1=this_barrier1,
        barrier2=this_barrier2,
        pd=this_pd,
        pl=this_pl,
        pu=this_pu
    )
    compare_td_and_mc(
        fmrp=random_walk,
        gamma=1.0,
        mc_episode_length_tol=1e-6,
        num_episodes=700,
        learning_rates=[(0.01, 1e8, 0.5), (0.05, 1e8, 0.5)],
        initial_vf_dict={s: 0.5 for s in random_walk.non_terminal_states},
        plot_batch=7,
        plot_start=0
    )
