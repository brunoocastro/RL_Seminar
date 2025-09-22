import numpy as np
from tqdm import tqdm

from grid import EpsilonGreedy
from grid.settings.params import Params

from gymnasium import Env
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    plt = None
    _HAS_MATPLOTLIB = False

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

            # Decide whether this run should include a rendered episode
            render_this_run = params.render_each > 0 and (run + 1) % params.render_each == 0
            # We'll render only a single episode for the run (use the last episode of the run)
            render_episode_index = params.total_episodes - 1 if render_this_run else None

            for episode in tqdm(
                episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
            ):
                state = env.reset()[0]  # Reset the environment
                step = 0
                done = False
                total_rewards = 0
                # Only collect frames for a single episode in the run (the configured one)
                render_this_episode = render_episode_index is not None and episode == render_episode_index
                frames = []
                while not done:
               
                    action = self.explorer.choose_action(
                        action_space=env.action_space, state=state, qtable=self.qtable
                    )

                    # Log all states and actions
                    all_states.append(state)
                    all_actions.append(action)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_state, reward, terminated, truncated, info = env.step(action)
                    if render_this_episode:

                        # print(f"Step {step}: State {state}, Action {action}, Reward {reward}, New State {new_state}, Terminated {terminated}, Truncated {truncated}")
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

                # If we collected frames for this episode, play them back as a full episode
                if render_this_episode and len(frames) > 0:
                    print(f"Rendering episode {episode} of run {run}: {len(frames)} frames")
                    if _HAS_MATPLOTLIB:
                        try:
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.axis('off')
                            im = ax.imshow(frames[0])

                            # Draw each frame once from start to end, then close the window
                            for fr in frames:
                                im.set_data(fr)
                                fig.canvas.draw()
                                fig.canvas.flush_events()
                                # Pause a little so the user can see each frame
                                try:
                                    delay = params.render_delay if hasattr(params, "render_delay") else 0.05
                                    plt.pause(delay)
                                except Exception:
                                    plt.pause(0.05)
                            plt.close(fig)
                        except Exception:
                            # If incremental drawing fails, fall back to showing the last frame
                            try:
                                fig, ax = plt.subplots(figsize=(4, 4))
                                ax.axis('off')
                                ax.imshow(frames[-1])
                                plt.show()
                                plt.close(fig)
                            except Exception:
                                pass
                    else:
                        print("matplotlib not available: skipping full-episode render")
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