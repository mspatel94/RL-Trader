from abc import ABC, abstractmethod
from portfolio import PortfolioState
from agent import AgentState

class Policy(ABC):

    @abstractmethod
    def get_action(self, state: AgentState):
        pass
    