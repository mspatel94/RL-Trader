from abc import ABC, abstractmethod
from portfolio import PortfolioState

class Policy(ABC):

    @abstractmethod
    def get_action(self, state: PortfolioState):
        pass
    