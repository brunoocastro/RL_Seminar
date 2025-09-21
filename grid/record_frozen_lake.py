import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from grid.EpsilonGreedy import EpsilonGreedy
from grid.learning_techniques.QLearning import QLearning
from grid.settings.params import Params

params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=11,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.85,
)

env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(size=params.map_size, p=params.proba_frozen, seed=params.seed),
)

env = RecordVideo(
    env,
    video_folder="./videos",
    episode_trigger=lambda x: x / 10 == 0,  # Record every 10th episode
    name_prefix="frozenlake-trained-agent",
)

# Add episode statistics tracking
env = RecordEpisodeStatistics(env)

env.action_space.seed(params.seed)

explorer = EpsilonGreedy(
    epsilon=params.epsilon,
)
learner = QLearning(
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    explorer=explorer,
)

learner.load_qtable()


done = False
step = 0
while not done:
    if step == 0:
        state, info = env.reset(seed=params.seed)
        print("Initial State:")
        print(state)
    
    step += 1

    action = learner.act(
      state
    )
    print(f"[{step}] State: {state}, Action: {action}")

    new_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    if done:
        print(f"Final State: {new_state}, Reward: {reward}")
        print(f"Finished due to {'termination' if terminated else 'truncation'} after {step} steps.")
        break
    else:
        state = new_state

    env.render()
env.close()


