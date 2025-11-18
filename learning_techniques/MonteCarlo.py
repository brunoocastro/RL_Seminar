import numpy as np
from tqdm import tqdm

from explorer_strategies import ExplorerStrategy
from params import TrainingConfig

from gymnasium import Env

try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except Exception:
    plt = None
    _HAS_MATPLOTLIB = False


class MonteCarlo:
    """Monte Carlo Control algorithm with First-Visit updates.

    Monte Carlo (MC) methods learn from complete episodes by calculating actual returns
    (sum of discounted rewards). Unlike Temporal Difference (TD) methods like Q-Learning
    or SARSA, Monte Carlo does not bootstrap - it waits for the full episode to finish
    and uses the actual experienced returns rather than value estimates.

    Key Characteristics:
    - Learns from complete episodes (episodic tasks only)
    - No bootstrapping - uses actual returns G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    - First-Visit MC: Updates Q(s,a) only on the first occurrence of (s,a) in an episode
    - Updates happen after episode completion, not during steps
    - Model-free: Does not require knowledge of environment dynamics

    Differences from TD Methods:
    - Q-Learning/SARSA: Update after each step using bootstrap estimates
    - Monte Carlo: Update after complete episode using actual returns

    This makes MC particularly effective for:
    - Episodic tasks with clear terminal states
    - Situations where bootstrapping might be unreliable
    - Learning without environment models

    Cannot be used for:
    - Continuing (non-episodic) tasks
    - Tasks where episodes are extremely long or infinite

    Attributes:
        learning_rate (float): Learning rate (alpha) for Q-value updates.
        gamma (float): Discount factor for future rewards.
        explorer (ExplorerStrategy): Exploration strategy for action selection.
    """

    def __init__(self, learning_rate: float, gamma: float, explorer: ExplorerStrategy):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.explorer = explorer

    def update(self, state, action, reward, new_state):
        """Stub method for API consistency with other learning techniques.

        Note: Monte Carlo does not perform step-by-step updates like TD methods.
        Instead, it collects complete episodes and updates using _update_from_episode().
        This method is included for API compatibility but is not used during training.

        Args:
            state: Current state (not used)
            action: Action taken (not used)
            reward: Reward received (not used)
            new_state: Next state (not used)

        Returns:
            float: 0.0 (no update performed)
        """
        # Monte Carlo updates happen after complete episodes, not individual steps
        return 0.0

    def _update_from_episode(self, states, actions, rewards):
        """Update Q-table using First-Visit Monte Carlo.

        This method implements the core Monte Carlo learning update. For each
        state-action pair (s,a) encountered in the episode:

        1. Find the first occurrence of (s,a) in the episode
        2. Calculate the actual return G from that point forward:
           G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^(T-t-1)R_T
        3. Update Q(s,a) toward the calculated return:
           Q(s,a) ← Q(s,a) + α[G - Q(s,a)]

        First-Visit vs Every-Visit:
        - First-Visit MC (implemented here): Only the first occurrence of (s,a)
          in an episode is used for updates. This is the classic MC approach.
        - Every-Visit MC (not implemented): Every occurrence of (s,a) would be used.

        Args:
            states: List of states visited in the episode
            actions: List of actions taken in the episode
            rewards: List of rewards received in the episode
        """
        # Track which (state, action) pairs we've already updated in this episode
        visited_state_actions = set()

        # Iterate through each time step in the episode
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            state_action = (state, action)

            # First-visit MC: skip if we've already seen this (s,a) pair
            if state_action in visited_state_actions:
                continue
            visited_state_actions.add(state_action)

            # Calculate the return G from time t onwards
            # G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
            G = 0
            for i in range(t, len(rewards)):
                G += (self.gamma ** (i - t)) * rewards[i]

            # Update Q(s,a) toward the actual return G
            # Q(s,a) ← Q(s,a) + α[G - Q(s,a)]
            self.qtable[state, action] += self.learning_rate * (
                G - self.qtable[state, action]
            )

    def reset_qtable(self, env: Env = None):
        """Reset the Q-table to zeros.

        Args:
            env: Gymnasium environment (optional). If provided, initializes
                 state_size and action_size from the environment.
        """
        if env is not None:
            self.state_size = env.observation_space.n
            self.action_size = env.action_space.n
        else:
            if self.state_size is None or self.action_size is None:
                raise ValueError("State size and action size must be defined.")

        self.qtable = np.zeros((self.state_size, self.action_size))

    def train(self, env: Env, config: TrainingConfig):
        """Train the Monte Carlo agent.

        The training loop follows the Monte Carlo pattern:
        1. Collect a complete episode (states, actions, rewards)
        2. After episode terminates, calculate returns and update Q-table
        3. Repeat for specified number of episodes and runs

        This is fundamentally different from TD methods which update during the episode.
        Monte Carlo must wait for episode completion to know the actual returns.

        Args:
            env: Gymnasium environment (must support episodic tasks)
            config: TrainingConfig instance with training parameters including:
                - total_episodes: Number of episodes per run
                - n_runs: Number of independent runs
                - render_each: Render every nth run (0 to disable)
                - render_delay: Delay between frames when rendering

        Returns:
            tuple: (rewards, steps, episodes, qtables, all_states, all_actions)
                - rewards: Array of total rewards per episode per run
                - steps: Array of steps taken per episode per run
                - episodes: Array of episode indices
                - qtables: Final Q-tables for each run
                - all_states: List of all states visited
                - all_actions: List of all actions taken
        """
        if env is None:
            raise ValueError(
                "Environment not found. Create environment using gym.make()."
            )

        self.reset_qtable(env=env)

        rewards = np.zeros((config.total_episodes, config.n_runs))
        steps = np.zeros((config.total_episodes, config.n_runs))
        episodes = np.arange(config.total_episodes)
        qtables = np.zeros((config.n_runs, self.state_size, self.action_size))
        all_states = []
        all_actions = []

        for run in range(
            config.n_runs
        ):  # Run several times to account for stochasticity
            self.reset_qtable(env=env)  # Reset the Q-table between runs

            # Decide whether this run should include a rendered episode
            render_this_run = (
                config.render_each > 0 and (run + 1) % config.render_each == 0
            )
            # We'll render only a single episode for the run (use the last episode of the run)
            render_episode_index = (
                config.total_episodes - 1 if render_this_run else None
            )

            for episode in tqdm(
                episodes, desc=f"Run {run}/{config.n_runs} - Episodes", leave=False
            ):
                # Episode storage for Monte Carlo updates
                episode_states = []
                episode_actions = []
                episode_rewards = []

                state = env.reset()[0]  # Reset the environment
                step = 0
                done = False
                total_rewards = 0
                # Only collect frames for a single episode in the run (the configured one)
                render_this_episode = (
                    render_episode_index is not None and episode == render_episode_index
                )
                frames = []

                # Collect complete episode trajectory
                while not done:
                    action = self.explorer.choose_action(
                        action_space=env.action_space, state=state, qtable=self.qtable
                    )

                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, terminated, truncated, info = env.step(action)

                    # Store trajectory for Monte Carlo update
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)

                    if render_this_episode:
                        print(f"Reward: {reward}, Cumulative Reward: {total_rewards}")

                        # Attempt to get an RGB array from the environment. Gym/Gymnasium render APIs differ,
                        # so try a couple of approaches and fall back if none work.
                        frame = None
                        try:
                            # Some envs accept a `mode` argument
                            frame = env.render(mode="rgb_array")
                        except TypeError:
                            try:
                                # Gymnasium often returns the frame directly when render() is called
                                frame = env.render()
                            except Exception:
                                frame = None
                        except Exception:
                            frame = None

                        if frame is not None:
                            frames.append(frame)

                    done = terminated or truncated

                    total_rewards += reward
                    step += 1

                    # Move to next state
                    state = new_state

                # Episode complete - now perform Monte Carlo updates
                self._update_from_episode(
                    episode_states, episode_actions, episode_rewards
                )

                # Log all rewards and steps
                rewards[episode, run] = total_rewards
                steps[episode, run] = step

                # If we collected frames for this episode, play them back as a full episode
                if render_this_episode and len(frames) > 0:
                    print(
                        f"Rendering episode {episode} of run {run}: {len(frames)} frames"
                    )
                    if _HAS_MATPLOTLIB:
                        try:
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.axis("off")
                            im = ax.imshow(frames[0])

                            # Draw each frame once from start to end, then close the window
                            for fr in frames:
                                im.set_data(fr)
                                fig.canvas.draw()
                                fig.canvas.flush_events()
                                # Pause a little so the user can see each frame
                                try:
                                    plt.pause(config.render_delay)
                                except Exception:
                                    plt.pause(0.05)
                            plt.close(fig)
                        except Exception:
                            # If incremental drawing fails, fall back to showing the last frame
                            try:
                                fig, ax = plt.subplots(figsize=(4, 4))
                                ax.axis("off")
                                ax.imshow(frames[-1])
                                plt.show()
                                plt.close(fig)
                            except Exception:
                                pass
                    else:
                        print("matplotlib not available: skipping full-episode render")
            qtables[run, :, :] = self.qtable

        return rewards, steps, episodes, qtables, all_states, all_actions

    def _table_name_from_prefix(self, prefix, size=(4, 4)):
        """Generate filename from prefix and grid size.

        Args:
            prefix: Filename prefix
            size: Grid size tuple (rows, cols)

        Returns:
            str: Formatted filename
        """
        return f"{prefix}_{size[0]}x{size[1]}.npy"

    def save_qtable(self, size=(4, 4), name_prefix="montecarlo_qtable"):
        """Save the Q-table to a file.

        Args:
            size: Grid size tuple (rows, cols)
            name_prefix: Filename prefix (default: "montecarlo_qtable")
        """
        filename = self._table_name_from_prefix(name_prefix, size)
        print(f"Saving Q-table to {filename}")
        np.save(filename, self.qtable)

    def load_qtable(self, size=(4, 4), name_prefix="montecarlo_qtable"):
        """Load the Q-table from a file.

        Args:
            size: Grid size tuple (rows, cols)
            name_prefix: Filename prefix (default: "montecarlo_qtable")
        """
        full_filename = self._table_name_from_prefix(name_prefix, size)
        print(f"Loading Q-table from {full_filename}")
        self.qtable = np.load(full_filename)

    def act(self, state):
        """Choose the best action for a given state based on learned Q-values.

        This is the greedy policy derived from the Q-table, used after training
        to exploit the learned policy.

        Args:
            state: Current state

        Returns:
            int: Action with highest Q-value for the given state
        """
        return np.argmax(self.qtable[state, :])
