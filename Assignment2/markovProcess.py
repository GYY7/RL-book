from abc import ABC, abstractmethod
from distribution import Distribution, FiniteDistribution
from state import State

class MarkovProcess(ABC, State):

    @abstractmethod
    def transition(self, state: State):# -> Optional[Distribution[State]]:
        pass

    def is_terminal(self, state: State) -> bool:
        return self.transition(state) is None

    def simulate(self, start_state_distribution: Distribution):
        state = start_state_distribution.sample()
        trace = [state]
        while not self.is_terminal(state):
            state_distribution = self.transition(state)
            state = state_distribution.sample()  # find the next state
            trace.append(state)
        return trace

class FiniteMarkovProcess(MarkovProcess):

    def __init__(self, transition_map: dict):
        self.non_terminal_states = [s for s, v in transition_map.items() if v is not None]
        self.transition_map = transition_map

    def transition(self, state: State):
        return self.transition_map[state]

    def states(self) -> list:
        return list(self.transition_map.keys())


