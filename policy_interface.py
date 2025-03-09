from abc import ABC, abstractmethod
from agent import AgentState

class Policy(ABC):

    @abstractmethod
    def get_action(self, state: AgentState):
        pass

    @abstractmethod
    def train(self, trajectory:AgentState):
        pass
    