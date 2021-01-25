from abc import ABC, abstractmethod
import scipy.stats as st
import numpy as np

# finite, infinite, discrete, continuous?
class Distribution:

    @abstractmethod
    def sample(self):
        pass


class FiniteDistribution(Distribution):

    def __init__(self, probabilities, values):
        self.values = values
        self.probabilities = probabilities

    def sample(self):
        return np.random.choice(self.values, p=self.probabilities)
