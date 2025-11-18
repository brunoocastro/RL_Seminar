from abc import ABC, abstractmethod
import numpy as np


class ExplorerStrategy(ABC):
    """Abstract base class for exploration strategies in reinforcement learning.
    
    An explorer strategy determines how an agent chooses actions during training,
    balancing exploration (trying new actions) and exploitation (using known good actions).
    
    Subclasses must implement the choose_action method to define their specific
    exploration-exploitation strategy.
    """
    
    def __init__(self, rng=np.random.default_rng()):
        """Initialize the explorer strategy.
        
        Args:
            rng: Random number generator (default: numpy's default_rng())
        """
        self.rng = rng
    
    @abstractmethod
    def choose_action(self, action_space, state, qtable):
        """Choose an action given the current state and Q-table.
        
        This method must be implemented by all explorer strategy subclasses.
        
        Args:
            action_space: The action space from the environment
            state: The current state
            qtable: The current Q-table containing learned values
            
        Returns:
            int: The chosen action
        """
        pass

    @abstractmethod
    def is_exploration(self):
        "Returns a boolean to represent if it should be a exploration step"
        pass

    @abstractmethod
    def decay(self):
        """
        Decrease based on the param the probability of exploration along the episodes
        """