from distribution import *
from markovProcess import *
from state import *
import scipy.stats as st
import numpy as np

# construct all the states
all_states = []
for i in range(101):
    all_states.append(State(i))

# define the snakes and ladders
snake_indices = [1, 6, 8, 14, 17, 34, 37, 50, 54, 63]
snake_to_indices = [38, 31, 49, 65, 53, 70, 76, 88, 98, 82]
snake_states = [all_states[i] for i in snake_indices]
snake_to_states = [all_states[i] for i in snake_to_indices]
ladder_indices = [35, 39, 41, 48, 51, 57, 74, 83, 85, 90, 92]
ladder_to_indices = [28, 3, 20, 7, 12, 25, 45, 77, 60, 67, 69]
ladder_states = [all_states[i] for i in ladder_indices]
ladder_to_states = [all_states[i] for i in ladder_to_indices]

# construct the transition map
transition_map = {}
for i in range(101):
    # if i in snake_indices:
    #     transition_map[all_states[i]] = FiniteDistribution(
    #         [1], [snake_to_states[snake_indices.index(i)]])
    # elif i in ladder_indices:
    #     transition_map[all_states[i]] = FiniteDistribution(
    #         [1], [ladder_to_states[ladder_indices.index(i)]])
    if i == 100:
        transition_map[all_states[i]] = None
    elif i > 94:
        prob_tmp = [1/6] * (100 - i)
        prob_tmp[100-i-1] += (6 - 100 + i)/6
        states_tmp = []
        for j in range(i + 1, 101):
            if j in snake_indices:
                states_tmp.append(snake_to_states[snake_indices.index(j)])
            else:
                states_tmp.append(all_states[j])
        transition_map[all_states[i]] = FiniteDistribution(
            prob_tmp, states_tmp
        )
    else:
        states_tmp = []
        for j in range(i + 1, i + 7):
            if j in snake_indices:
                states_tmp.append(snake_to_states[snake_indices.index(j)])
            else:
                states_tmp.append(all_states[j])
        transition_map[all_states[i]] = FiniteDistribution(
            [1/6]*6, states_tmp
        )

simulation = FiniteMarkovProcess(transition_map)


for state in simulation.simulate(transition_map[all_states[0]]):
    print(state.get_position())


