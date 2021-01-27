from rl.distribution import Categorical
from dataclasses import dataclass
from rl.markov_process import *
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlayerState:
    """
    Player state
    """
    position: int


class SnakesAndLaddersMPFinite(FiniteMarkovProcess[PlayerState]):
    """
    Finite Markov Process of Snakes and Ladders
    """
    def __init__(self, snake_positions: list, snake_to_positions: list,
                 ladder_positions: list, ladder_to_positions: list):
        self.snake_positions = snake_positions
        self.snake_to_positions = snake_to_positions
        self.ladder_positions = ladder_positions
        self.ladder_to_positions = ladder_to_positions
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Transition[PlayerState]:
        """
        return transition map
        """
        d: Dict[PlayerState, Optional[Categorical[PlayerState]]] = {}
        for i in range(101):
            states_prob_map: Mapping[PlayerState, float] = {}
            end = min(i+7, 101)
            prob = 1/6
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
            if i == 100:
                d[PlayerState(100)] = None
        return d


class SnakesAndLaddersMRPFinite(FiniteMarkovRewardProcess[PlayerState]):
    """
        Finite Markov Reward Process of Snakes and Ladders
        """
    def __init__(self, snake_positions, snake_to_positions, ladder_positions, ladder_to_positions):
        self.snake_positions = snake_positions
        self.snake_to_positions = snake_to_positions
        self.ladder_positions = ladder_positions
        self.ladder_to_positions = ladder_to_positions
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> RewardTransition[PlayerState]:
        """
        return transition reward map
        """
        d: Dict[PlayerState, Optional[Categorical[Tuple[PlayerState, int]]]] = {}
        for i in range(101):
            states_prob_map: Mapping[Tuple[PlayerState, int], float] = {}
            end = min(i + 7, 101)
            prob = 1 / 6
            for j in range(i + 1, end):
                if j == end - 1:
                    prob = 1 - (end - i - 2) / 6
                if j in self.snake_positions:
                    to_state_pos = self.snake_to_positions[self.snake_positions.index(j)]
                elif j in self.ladder_positions:
                    to_state_pos = self.ladder_to_positions[self.ladder_positions.index(j)]
                else:
                    to_state_pos = j
                states_prob_map[(PlayerState(to_state_pos), 1)] = prob
            d[PlayerState(i)] = Categorical(states_prob_map)
            if i == 100:
                d[PlayerState(100)] = None
        return d


if __name__ == '__main__':
    # construct the game
    snake_to_positions = [1, 6, 8, 14, 17, 34, 37, 50, 42, 54, 63]
    snake_positions = [38, 31, 49, 65, 53, 70, 76, 88, 94, 98, 82]
    ladder_to_positions = [35, 39, 41, 48, 51, 57, 74, 83, 85, 90, 92]
    ladder_positions = [28, 3, 20, 7, 12, 25, 45, 77, 60, 67, 69]
    sl_mp = SnakesAndLaddersMPFinite(snake_positions, snake_to_positions, ladder_positions, ladder_to_positions)
    sl_mrp = SnakesAndLaddersMRPFinite(snake_positions, snake_to_positions, ladder_positions, ladder_to_positions)

    print('Expected number of dices calculated by value function:',
          sl_mrp.get_value_function_vec(1)[0])

    # simulate
    transition_map = sl_mp.get_transition_map()
    traces_steps = []
    count = 0
    for trace in sl_mp.traces(transition_map[PlayerState(0)]):
        step = 0
        count += 1
        for pos in trace:
            step += 1
            if pos == PlayerState(100):
                traces_steps.append(step)
        if count == 100000:
            break

    # plot the probability distribution
    # x = list(set(traces_steps))
    # y = [traces_steps.count(step)/50000 for step in x]
    plt.hist(traces_steps, bins=20)
    plt.show()

    # Compare the expected number of dices calculated by simulation
    mean = 0
    for step in set(traces_steps):
        step_freq = traces_steps.count(step)/100000
        mean += step*step_freq
    print('Simulated expected number of dices:', mean)

