import numpy as np
from explorer_strategies.base import ExplorerStrategy


class EpsilonGreedy(ExplorerStrategy):
    """Epsilon-Greedy exploration strategy.
    
    This strategy balances exploration and exploitation using an epsilon parameter:
    - With probability epsilon: explore by choosing a random action
    - With probability (1-epsilon): exploit by choosing the action with highest Q-value
    
    This is one of the most common exploration strategies in reinforcement learning,
    providing a simple yet effective way to balance learning new information
    (exploration) with using existing knowledge (exploitation).
    
    Attributes:
        epsilon (float): Probability of choosing a random action (0 to 1)
        decay_rate (float): The amount we decrease the epsilon along episodes
        rng: Random number generator for action selection
    """
    
    def __init__(self, epsilon=1, decay_rate=0.001 ,rng=np.random.default_rng()):
        """Initialize the Epsilon-Greedy strategy.
        
        Args:
            epsilon (float): Exploration probability (0 to 1)
            rng: Random number generator (default: numpy's default_rng())
        """
        super().__init__(rng)
        self.epsilon = epsilon
        self.epsilon_decay_rate = decay_rate


    def is_exploration(self):
         # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        return explor_exploit_tradeoff < self.epsilon
    
    def decay(self):
        self.epsilon = max(0, self.epsilon - self.epsilon_decay_rate) # Decay epsilon but ensure it doesn't go below 0
        print(f"- Epsilon after decay: {self.epsilon}")


    def choose_action(self, action_space, state, qtable):
        """Choose an action using epsilon-greedy strategy.
        
        Args:
            action_space: The action space from the environment
            state: The current state
            qtable: The current Q-table containing learned values
            
        Returns:
            int: The chosen action
        """
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # Find the indices where the Q-value equals the maximum value
            # Choose a random action from the indices where the Q-value is maximum
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = self.rng.choice(max_ids)
        return action