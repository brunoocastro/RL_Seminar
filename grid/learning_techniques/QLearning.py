import numpy as np
from tqdm import tqdm

from grid import EpsilonGreedy
from grid.settings.params import Params

from gymnasium import Env

class QLearning:
    """Q-learning algorithm.
    
    Attributes:
        learning_rate (float): Learning rate.
        gamma (float): Discount factor.
        explorer (EpsilonGreedy): Exploration strategy.
    """
    def __init__(self, learning_rate:float, gamma:float, explorer:EpsilonGreedy):

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.explorer = explorer

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self, env: Env = None):
        """Reset the Q-table."""
        if env is not None:
            self.state_size = env.observation_space.n
            self.action_size = env.action_space.n
        else: 
            if self.state_size is None or self.action_size is None:
                raise ValueError("State size and action size must be defined.")
            
        self.qtable = np.zeros((self.state_size, self.action_size))

    def train(self, env: Env, params: Params):
        if env is None:
            raise ValueError("Environment not found. Create environment using gym.make().")

        self.reset_qtable(env=env)


        rewards = np.zeros((params.total_episodes, params.n_runs))
        steps = np.zeros((params.total_episodes, params.n_runs))
        episodes = np.arange(params.total_episodes)
        qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
        all_states = []
        all_actions = []

        

        for run in range(params.n_runs):  # Run several times to account for stochasticity
            self.reset_qtable(env=env)  # Reset the Q-table between runs



            for episode in tqdm(
                episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
            ):
                state = env.reset(seed=params.seed)[0]  # Reset the environment
                step = 0
                done = False
                total_rewards = 0

                render_this_episode = params.render_each > 0 and (episode + 1) % params.render_each == 0
                while not done:
                    if render_this_episode:
                        env.render()
               
                    action = self.explorer.choose_action(
                        action_space=env.action_space, state=state, qtable=self.qtable
                    )

                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, terminated, truncated, info = env.step(action)

                    done = terminated or truncated

                    self.qtable[state, action] = self.update(
                        state, action, reward, new_state
                    )

                    total_rewards += reward
                    step += 1

                    # Our new state is state
                    state = new_state

                # if params.verbose and (episode + 1) % 100 == 0:
                #     print(
                #         f"Run: {run+1}/{params.n_runs}, Episode: {episode+1}/{params.total_episodes}, Steps: {step}, Total Reward: {total_rewards}"
                #     )

                # Log all rewards and steps
                rewards[episode, run] = total_rewards
                steps[episode, run] = step
            qtables[run, :, :] = self.qtable

        return rewards, steps, episodes, qtables, all_states, all_actions
    

    def save_qtable(self, size= (5, 5),  name_prefix = "qlearning_qtable.npy"):
        """Save the Q-table to a file."""
        filename = f"{name_prefix}_{size[0]}x{size[1]}.npy"
        print(f"Saving Q-table to {filename}")
        np.save(filename, self.qtable)

    def load_qtable(self, filename = "qlearning_qtable.npy"):
        """Load the Q-table from a file."""
        print(f"Loading Q-table from {filename}")
        self.qtable = np.load(filename)

    def act(self, state):
        """Choose the best action for a given state."""
        return np.argmax(self.qtable[state, :])