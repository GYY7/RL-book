from rl.distribution import Categorical
from dataclasses import dataclass
from rl.markov_process import *
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class FrogState:
    position: int

    def frog_position(self) -> int:
        return self.position


class FrogPuzzleMPFinite(FiniteMarkovProcess[FrogState]):

    def __init__(self, num_leaves):
        self.num_leaves = num_leaves
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Transition[FrogState]:
        d: Dict[FrogState, Optional[Categorical[FrogState]]] = {}
        for i in range(self.num_leaves):
            states_prob_map: Mapping[FrogState, float] = {}
            for j in range(i + 1, self.num_leaves + 1):
                states_prob_map[FrogState(j)] =  1/(self.num_leaves - i + 1)
            d[FrogState(i)] = Categorical(states_prob_map)
        d[FrogState(self.num_leaves)] = None
        return d


if __name__ == '__main__':
    fp_mp = FrogPuzzleMPFinite(10)
    # print(fp_mp)
    transition_map = fp_mp.get_transition_map()
    traces_steps = []
    count = 0
    for trace in fp_mp.traces(transition_map[FrogState(0)]):
        step = 0
        count += 1
        for pos in trace:
            step += 1
            if pos == FrogState(10):
                traces_steps.append(step)
        if count == 10000:
            break
    plt.hist(traces_steps, bins=20)
    plt.show()

    mean = 0
    for step in set(traces_steps):
        step_freq = traces_steps.count(step) / 10000
        mean += step * step_freq
    print(mean)

