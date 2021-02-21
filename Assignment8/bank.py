from rl import *
from rl.markov_decision_process import FiniteMarkovDecisionProcess, StateActionMapping, FinitePolicy, FiniteMarkovRewardProcess
from dataclasses import dataclass
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar, Any)
from rl.distribution import Categorical, Constant
import itertools
import numpy as np
import matplotlib.pyplot as plt
from rl.dynamic_programming import *


@dataclass(frozen=True)
class BankState:
    cash: int


BankMapping = StateActionMapping[BankState, float]
# actions: alpha(portion of investment)
# deposits and withdraws?

class BankStrategy(FiniteMarkovDecisionProcess[BankState]):

