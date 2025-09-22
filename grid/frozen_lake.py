from grid.EpsilonGreedy import EpsilonGreedy
from grid.learning_techniques.QLearning import QLearning
from grid.settings.params import Params

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gymnasium as gym

from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


params = Params(
    total_episodes=2000,
    learning_rate=0.7,
    gamma=0.95,
    epsilon=0.1,
    map_size=11,
    seed=123,
    is_slippery=False,
    n_runs=2000,
    action_size=None,
    state_size=None,
    proba_frozen=0.85,
)


def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


# %%
# We want to plot the policy the agent has learned in the end. To do that
# we will: 1. extract the best Q-values from the Q-table for each state,
# 2. get the corresponding best action for those Q-values, 3. map each
# action to an arrow so we can visualize it.
#


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


# %%
# With the following function, we'll plot on the left the last frame of
# the simulation. If the agent learned a good policy to solve the task, we
# expect to see it on the tile of the treasure in the last frame of the
# video. On the right we'll plot the policy the agent has learned. Each
# arrow will represent the best action to choose for each tile/state.
#


def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()


def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    plt.show()


##################################

explorer = EpsilonGreedy(
    epsilon=params.epsilon,
)
learner = QLearning(
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    explorer=explorer,
)

# map_sizes = [4, 7, 9, 11]
map_sizes = [11]
res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",  # Use rgb_array mode for controlled rendering
        desc=generate_random_map(
            size=map_size, p=params.proba_frozen, seed=params.seed
        ),
    )

    env = RecordVideo(
        env,
        video_folder="./videos/frozenlake",
        episode_trigger=lambda x: x / params.render_each == 0,  # Record every n episodes
        name_prefix="fl-training",
    )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env)

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    )  # Set the seed to get reproducible results when sampling the action space

    print(f"Map size: {map_size}x{map_size}")
    rewards, steps, episodes, qtables, all_states, all_actions = learner.train(
        env, params
    )

    # Save the results in dataframes
    res, st = postprocess(episodes, params, rewards, steps, map_size)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    qtable = qtables.mean(axis=0)  # Average the Q-table between runs

    # plot_states_actions_distribution(
    #     states=all_states, actions=all_actions, map_size=map_size
    # )  # Sanity check
    # plot_q_values_map(qtable, env, map_size)
    learner.save_qtable(
        (env.action_space.n, env.observation_space.n), name_prefix="qlearning_qtable"
    )
    env.close()


def plot_steps_and_rewards(rewards_df, steps_df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    plt.show()


# plot_steps_and_rewards(res_all, st_all)
