from abc import ABC, abstractmethod
from agent import AgentState
import numpy as np
class Policy(ABC):

    @abstractmethod
    def get_action(self, state: AgentState)->np.ndarray:
        pass

    @abstractmethod
    def train(self, trajectory:AgentState):
        pass
    