from rl.distribution import Categorical
from dataclasses import dataclass
from rl.markov_process import *
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlayerState:
    position: int

    def player_position(self) -> int:
        return self.position

class SnakesAndLaddersMPFinite(FiniteMarkovProcess[PlayerState]):

    def __init__(self, snake_positions, snake_to_positions, ladder_positions, ladder_to_positions):
        self.snake_positions = snake_positions
        self.snake_to_positions = snake_to_positions
        self.ladder_positions = ladder_positions
        self.ladder_to_positions = ladder_to_positions
        super().__init__(self.get_transition_map())

    def get_transition_map(self) \
    -> Transition[PlayerState]:
        d: Dict[PlayerState, Categorical[PlayerState]] = {}
        for i in range(101):
            states_prob_map: Mapping[PlayerState, float] = {}
            end = min(i+7, 101)
            prob = 1/6
            if i == 100:
                d[PlayerState(100)] = None
            for j in range(i+1, end):
                if j == end - 1:
                    prob = 1 - (end - i - 2) / 6
                if j in self.snake_positions:
                    states_prob_map[PlayerState(self.snake_to_positions[self.snake_positions.index(j)])] = prob
                elif j in self.ladder_positions:
                    states_prob_map[PlayerState(self.ladder_to_positions[self.ladder_positions.index(j)])] = prob
                else:
                    states_prob_map[PlayerState(j)] = prob
            d[PlayerState(i)] = Categorical(states_prob_map)
        return d


snake_to_positions = [1, 6, 8, 14, 17, 34, 37, 50, 42, 54, 63]
snake_positions = [38, 31, 49, 65, 53, 70, 76, 88, 94, 98, 82]
ladder_to_positions = [35, 39, 41, 48, 51, 57, 74, 83, 85, 90, 92]
ladder_positions = [28, 3, 20, 7, 12, 25, 45, 77, 60, 67, 69]
sl_mp = SnakesAndLaddersMPFinite(snake_positions, snake_to_positions, ladder_positions, ladder_to_positions)
# print(sl_mp)
transition_map = sl_mp.get_transition_map()
traces_steps = []
for k in range(10000):
    pos_lst = []
    try:
        for pos in sl_mp.simulate(transition_map[PlayerState(0)]):
            pos_lst.append(pos)
    except IndexError:
        traces_steps.append(len(pos_lst))

plt.hist(traces_steps, bins=20)
plt.show()
